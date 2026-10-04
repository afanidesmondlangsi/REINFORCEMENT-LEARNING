import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 1. LOAD CONDITION-LEVEL RESULTS
# ============================================================

df = pd.read_csv(
    "rq3_analysis/rq3_relatedness_performance.csv"
)

df["shift"] = pd.to_numeric(df["shift"])


# ============================================================
# 2. REMOVE THE DUPLICATE EFFECTIVE HOPPER CONDITION
#
# Hopper 0.1 and 0.5 were experimentally verified to produce
# the same effective floor-foot friction and identical results.
#
# Keep the original 16-condition CSV unchanged.
# Remove Hopper 0.5 only for this effective-condition analysis.
# ============================================================

effective = df[
    ~(
        (df["environment"] == "hopper-friction")
        & np.isclose(df["shift"], 0.5)
    )
].copy()

print("\nEffective conditions:")
print(len(effective))

print("\nConditions per environment:")
print(
    effective.groupby("environment")
    .size()
)


# ============================================================
# 3. VARIABLES TO CENTER WITHIN ENVIRONMENT
# ============================================================

variables = [
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


# ============================================================
# 4. WITHIN-ENVIRONMENT CENTERING
#
# For each variable:
#
# centered value =
# condition value - environment mean
#
# This removes differences in overall scale between
# Ant, HalfCheetah, Hopper and Walker2d.
# ============================================================

for variable in variables:

    effective[variable + "_centered"] = (
        effective[variable]
        - effective.groupby("environment")[variable]
        .transform("mean")
    )


# ============================================================
# 5. WITHIN-ENVIRONMENT Z-SCORES
#
# This additionally puts environments on a comparable scale.
#
# ddof=0 is used here for descriptive standardization.
# ============================================================

for variable in variables:

    group_mean = (
        effective.groupby("environment")[variable]
        .transform("mean")
    )

    group_sd = (
        effective.groupby("environment")[variable]
        .transform(
            lambda x: x.std(ddof=0)
        )
    )

    effective[variable + "_z"] = (
        (effective[variable] - group_mean)
        / group_sd
    )


# ============================================================
# 6. CORRELATION HELPER
# ============================================================

def calculate_correlation(data, x, y):

    temp = data[[x, y]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(temp) < 3:
        return None

    pearson_r, pearson_p = pearsonr(
        temp[x],
        temp[y]
    )

    spearman_rho, spearman_p = spearmanr(
        temp[x],
        temp[y]
    )

    return {
        "x": x,
        "y": y,
        "n": len(temp),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    }


# ============================================================
# 7. CENTERED ANALYSIS
# ============================================================

relationships_centered = [
    (
        "mean_distance_mean_centered",
        "bcpar_auc_centered"
    ),
    (
        "mean_distance_mean_centered",
        "bcpar_minus_bcsac_centered"
    ),
    (
        "mean_distance_mean_centered",
        "bcpar_minus_rlpd_centered"
    ),
    (
        "mean_distance_mean_centered",
        "bcpar_advantage_centered"
    ),
    (
        "final_distance_mean_centered",
        "bcpar_auc_centered"
    ),
    (
        "final_distance_mean_centered",
        "bcpar_advantage_centered"
    ),
    (
        "mean_src_reward_mean_centered",
        "bcpar_auc_centered"
    ),
    (
        "mean_src_reward_mean_centered",
        "bcpar_advantage_centered"
    )
]

centered_results = []

for x, y in relationships_centered:

    result = calculate_correlation(
        effective,
        x,
        y
    )

    if result is not None:
        centered_results.append(result)

centered_results = pd.DataFrame(
    centered_results
)


# ============================================================
# 8. Z-SCORE ANALYSIS
# ============================================================

relationships_z = [
    (
        "mean_distance_mean_z",
        "bcpar_auc_z"
    ),
    (
        "mean_distance_mean_z",
        "bcpar_minus_bcsac_z"
    ),
    (
        "mean_distance_mean_z",
        "bcpar_minus_rlpd_z"
    ),
    (
        "mean_distance_mean_z",
        "bcpar_advantage_z"
    ),
    (
        "final_distance_mean_z",
        "bcpar_auc_z"
    ),
    (
        "final_distance_mean_z",
        "bcpar_advantage_z"
    ),
    (
        "mean_src_reward_mean_z",
        "bcpar_auc_z"
    ),
    (
        "mean_src_reward_mean_z",
        "bcpar_advantage_z"
    )
]

z_results = []

for x, y in relationships_z:

    result = calculate_correlation(
        effective,
        x,
        y
    )

    if result is not None:
        z_results.append(result)

z_results = pd.DataFrame(
    z_results
)


# ============================================================
# 9. PER-ENVIRONMENT DESCRIPTIVE CORRELATIONS
#
# These are exploratory only because each environment has
# only 3 or 4 effective shift conditions.
# ============================================================

per_environment_rows = []

for environment, group in effective.groupby(
    "environment"
):

    if len(group) >= 3:

        for y in [
            "bcpar_auc",
            "bcpar_advantage"
        ]:

            pearson_r, pearson_p = pearsonr(
                group["mean_distance_mean"],
                group[y]
            )

            spearman_rho, spearman_p = spearmanr(
                group["mean_distance_mean"],
                group[y]
            )

            per_environment_rows.append({
                "environment": environment,
                "outcome": y,
                "n_conditions": len(group),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p
            })

per_environment = pd.DataFrame(
    per_environment_rows
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

effective.to_csv(
    "rq3_analysis/"
    "rq3_effective_conditions_within_environment.csv",
    index=False
)

centered_results.to_csv(
    "rq3_analysis/"
    "rq3_within_environment_centered_correlations.csv",
    index=False
)

z_results.to_csv(
    "rq3_analysis/"
    "rq3_within_environment_z_correlations.csv",
    index=False
)

per_environment.to_csv(
    "rq3_analysis/"
    "rq3_per_environment_correlations.csv",
    index=False
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n================================================")
print("WITHIN-ENVIRONMENT CENTERED CORRELATIONS")
print("================================================")

print(
    centered_results.to_string(
        index=False
    )
)

print("\n================================================")
print("WITHIN-ENVIRONMENT Z-SCORE CORRELATIONS")
print("================================================")

print(
    z_results.to_string(
        index=False
    )
)

print("\n================================================")
print("PER-ENVIRONMENT DESCRIPTIVE CORRELATIONS")
print("CAUTION: ONLY 3-4 CONDITIONS PER ENVIRONMENT")
print("================================================")

print(
    per_environment.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    "rq3_analysis/"
    "rq3_effective_conditions_within_environment.csv"
)
print(
    "rq3_analysis/"
    "rq3_within_environment_centered_correlations.csv"
)
print(
    "rq3_analysis/"
    "rq3_within_environment_z_correlations.csv"
)
print(
    "rq3_analysis/"
    "rq3_per_environment_correlations.csv"
)