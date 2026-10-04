import pandas as pd
import numpy as np
from scipy.stats import t

# ============================================================
# Load per-run normalized AUC results
# ============================================================

df = pd.read_csv(
    "rq3_analysis/rq3_auc_by_run.csv"
)

metric = "normalized_auc_5000_10000"

# ============================================================
# Summarize across seeds
# ============================================================

summary = (
    df.groupby(
        ["environment", "shift", "algorithm"]
    )[metric]
    .agg(
        mean="mean",
        sd="std",
        median="median",
        n="count"
    )
    .reset_index()
)

# ============================================================
# Standard error
# ============================================================

summary["se"] = (
    summary["sd"] /
    np.sqrt(summary["n"])
)

# ============================================================
# 95% Student-t confidence interval
# ============================================================

summary["t_critical"] = summary["n"].apply(
    lambda n: t.ppf(0.975, df=n - 1)
)

summary["ci95_half_width"] = (
    summary["t_critical"] *
    summary["se"]
)

summary["ci95_lower"] = (
    summary["mean"] -
    summary["ci95_half_width"]
)

summary["ci95_upper"] = (
    summary["mean"] +
    summary["ci95_half_width"]
)

# ============================================================
# Determine best mean AUC within each environment × shift
# ============================================================

summary["best_mean_auc"] = (
    summary.groupby(
        ["environment", "shift"]
    )["mean"]
    .transform("max")
)

summary["winner"] = np.isclose(
    summary["mean"],
    summary["best_mean_auc"]
)

# ============================================================
# Sort
# ============================================================

algorithm_order = {
    "BC_SAC": 0,
    "RLPD": 1,
    "BC_PAR": 2
}

summary["algorithm_order"] = (
    summary["algorithm"]
    .map(algorithm_order)
)

summary = (
    summary
    .sort_values(
        [
            "environment",
            "shift",
            "algorithm_order"
        ]
    )
    .drop(
        columns=[
            "algorithm_order",
            "best_mean_auc",
            "t_critical"
        ]
    )
)

# ============================================================
# Save
# ============================================================

output = (
    "rq3_analysis/"
    "rq3_auc_summary_with_ci.csv"
)

summary.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)

print("\nNumber of conditions:")
print(
    summary[
        ["environment", "shift"]
    ]
    .drop_duplicates()
    .shape[0]
)

print("\nNumber of algorithm summaries:")
print(len(summary))

print("\nWinner counts:")
print(
    summary[
        summary["winner"]
    ]["algorithm"]
    .value_counts()
)

print("\nAUC SUMMARY WITH 95% CI")
print(
    summary[
        [
            "environment",
            "shift",
            "algorithm",
            "mean",
            "sd",
            "ci95_lower",
            "ci95_upper",
            "winner"
        ]
    ].to_string(index=False)
)