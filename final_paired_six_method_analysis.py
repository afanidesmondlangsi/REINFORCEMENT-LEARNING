import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# ============================================================
# FINAL PAIRED SIX-ALGORITHM STATISTICAL ANALYSIS
# ============================================================

ROOT = Path("matched_final_5000")
OUT = ROOT / "final_paired_six_method_analysis"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_UNNORMALIZED",
    "EA_NTC_NORMALIZED"
]

KEYS = ["environment", "shift", "seed"]

EVAL = [
    "eval_1000",
    "eval_2000",
    "eval_3000",
    "eval_4000",
    "eval_5000"
]

N_BOOT = 10000
RNG = np.random.default_rng(20261101)

# ------------------------------------------------------------
# 1. Load and validate
# ------------------------------------------------------------

print("\n========== FINAL PAIRED ANALYSIS ==========")

df = pd.read_csv(ROOT / "matched_final_288_long.csv")
df = df[df.method.isin(METHODS)].copy()

assert len(df) == 288
assert not df.duplicated(KEYS + ["method"]).any()
assert not df[EVAL].isna().any().any()

coverage = df.groupby(KEYS).method.nunique()
assert len(coverage) == 48
assert (coverage == 6).all()

steps = np.array([1000, 2000, 3000, 4000, 5000])

df["auc"] = np.trapz(
    df[EVAL].to_numpy(),
    x=steps,
    axis=1
)

df.to_csv(OUT / "validated_seed_level_results.csv", index=False)

print("Records:", len(df))
print("Matched conditions:", len(coverage))
print("Methods:", len(METHODS))

# ------------------------------------------------------------
# 2. Seed-level variability
# ------------------------------------------------------------

print("\n========== SEED-LEVEL VARIABILITY ==========")

seed_summary = (
    df.groupby(["environment", "shift", "method"])
    .agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        final_return_mean=("eval_5000", "mean"),
        final_return_std=("eval_5000", "std"),
        seed_count=("seed", "nunique")
    )
    .reset_index()
)

seed_summary.to_csv(
    OUT / "seed_level_variability.csv", index=False
)

# Environment-level summary
environment_summary = (
    df.groupby(["environment", "method"])
    .agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        final_return_mean=("eval_5000", "mean"),
        final_return_std=("eval_5000", "std"),
        observations=("seed", "count")
    )
    .reset_index()
)

environment_summary.to_csv(
    OUT / "environment_level_summary.csv", index=False
)

print(environment_summary.round(3).to_string(index=False))

# ------------------------------------------------------------
# 3. Stratified paired bootstrap
# ------------------------------------------------------------

def paired_bootstrap(data, metric, group_column=None):

    # Every experimental condition must be equally represented.
    if group_column is None:
        strata = ["environment", "shift"]
    else:
        strata = [group_column, "shift"]

    groups = [
        g[metric].to_numpy()
        for _, g in data.groupby(strata, sort=True)
    ]

    # Each stratum contains three matched seeds.
    assert all(len(g) == 3 for g in groups)

    observed = np.concatenate(groups).mean()

    samples = np.empty(N_BOOT)

    for b in range(N_BOOT):
        sampled = [
            RNG.choice(g, size=3, replace=True)
            for g in groups
        ]
        samples[b] = np.concatenate(sampled).mean()

    return (
        observed,
        np.percentile(samples, 2.5),
        np.percentile(samples, 97.5)
    )

# ------------------------------------------------------------
# 4. All 15 pairwise comparisons
# ------------------------------------------------------------

print("\n========== ALL PAIRWISE COMPARISONS ==========")

pairwise = []
env_pairwise = []

wide_auc = df.pivot(
    index=KEYS,
    columns="method",
    values="auc"
)

wide_final = df.pivot(
    index=KEYS,
    columns="method",
    values="eval_5000"
)

for method_a, method_b in combinations(METHODS, 2):

    for metric_name, wide in [
        ("AUC", wide_auc),
        ("Final_Return", wide_final)
    ]:

        differences = wide[method_a] - wide[method_b]

        temp = differences.rename("difference").reset_index()

        mean_diff, ci_low, ci_high = paired_bootstrap(
            temp, "difference"
        )

        pairwise.append({
            "metric": metric_name,
            "method_a": method_a,
            "method_b": method_b,
            "mean_difference": mean_diff,
            "std_difference": differences.std(),
            "win_percentage": 100 * (differences > 0).mean(),
            "tie_percentage": 100 * np.isclose(
                differences, 0
            ).mean(),
            "CI_2.5": ci_low,
            "CI_97.5": ci_high,
            "conditions": len(differences),
            "CI_excludes_zero": (
                ci_low > 0 or ci_high < 0
            )
        })

        # Environment-specific paired comparisons
        for env in sorted(df.environment.unique()):

            env_data = temp[
                temp.environment == env
            ].copy()

            env_diff = env_data["difference"]

            mean_env, low_env, high_env = paired_bootstrap(
                env_data,
                "difference",
                group_column="environment"
            )

            env_pairwise.append({
                "metric": metric_name,
                "environment": env,
                "method_a": method_a,
                "method_b": method_b,
                "mean_difference": mean_env,
                "std_difference": env_diff.std(),
                "win_percentage": 100 * (env_diff > 0).mean(),
                "CI_2.5": low_env,
                "CI_97.5": high_env,
                "conditions": len(env_diff),
                "CI_excludes_zero": (
                    low_env > 0 or high_env < 0
                )
            })

pair_df = pd.DataFrame(pairwise)
env_pair_df = pd.DataFrame(env_pairwise)

pair_df.to_csv(
    OUT / "all_15_paired_comparisons.csv", index=False
)

env_pair_df.to_csv(
    OUT / "environment_paired_comparisons.csv", index=False
)

print("\nOVERALL AUC COMPARISONS")

print(
    pair_df[pair_df.metric == "AUC"]
    .round(3)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 5. Focused EA-NTC comparisons against four baselines
# ------------------------------------------------------------

print("\n========== EA-NTC VS FOUR BASELINES ==========")

focused = pair_df[
    pair_df.method_a.isin([
        "EA_NTC_UNNORMALIZED",
        "EA_NTC_NORMALIZED"
    ]) &
    pair_df.method_b.isin([
        "SAC_TARGET_ONLY",
        "BC_SAC",
        "BC_PAR",
        "RLPD"
    ])
].copy()

focused.to_csv(
    OUT / "EA_NTC_vs_four_baselines_with_CI.csv",
    index=False
)

print(focused.round(3).to_string(index=False))

# ------------------------------------------------------------
# 6. Learning performance across checkpoints
# ------------------------------------------------------------

print("\n========== CHECKPOINT PERFORMANCE ==========")

checkpoint_summary = (
    df.groupby(["environment", "method"])[EVAL]
    .agg(["mean", "std"])
)

checkpoint_summary.to_csv(
    OUT / "checkpoint_mean_std.csv"
)

# ------------------------------------------------------------
# 7. Final report
# ------------------------------------------------------------

report = OUT / "statistical_analysis_summary.txt"

with open(report, "w", encoding="utf-8") as f:

    f.write("FINAL PAIRED SIX-ALGORITHM ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write("Records: 288\n")
    f.write("Matched conditions: 48\n")
    f.write("Methods: 6\n")
    f.write("Seeds: 3\n\n")

    f.write("AUC: trapezoidal integration from 1000 to 5000 steps.\n")
    f.write("Bootstrap: 10000 stratified paired resamples.\n")
    f.write("Strata: environment x friction shift.\n\n")

    f.write("ENVIRONMENT SUMMARY\n")
    f.write(environment_summary.to_string(index=False))
    f.write("\n\nPAIRWISE COMPARISONS\n")
    f.write(pair_df.to_string(index=False))
    f.write("\n\nENVIRONMENT PAIRWISE COMPARISONS\n")
    f.write(env_pair_df.to_string(index=False))

print("\n========== COMPLETED ==========")

print("Output directory:", OUT.resolve())

print("\nGenerated files:")
for p in sorted(OUT.glob("*")):
    print(p.name)

