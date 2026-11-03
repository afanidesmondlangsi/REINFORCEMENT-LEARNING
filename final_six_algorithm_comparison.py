import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FINAL SIX-ALGORITHM COMPARISON
# NO EA-NTC MECHANISTIC DIAGNOSTICS
# ============================================================

ROOT = Path("matched_final_5000")
OUT = ROOT / "final_six_algorithm_comparison"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "matched_final_288_long.csv")

methods = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_UNNORMALIZED",
    "EA_NTC_NORMALIZED"
]

metrics = [
    "eval_1000",
    "eval_2000",
    "eval_3000",
    "eval_4000",
    "eval_5000"
]

keys = ["environment", "shift", "seed"]

# ------------------------------------------------------------
# 1. Validate dataset
# ------------------------------------------------------------

print("\n========== DATA VALIDATION ==========")

df = df[df["method"].isin(methods)].copy()

assert len(df) == 288, f"Expected 288 records, found {len(df)}"

assert not df.duplicated(keys + ["method"]).any(), \
    "Duplicate environment/shift/seed/method records found"

assert not df[metrics].isna().any().any(), \
    "Missing evaluation metrics detected"

coverage = df.groupby(keys)["method"].nunique()

assert len(coverage) == 48
assert (coverage == 6).all()

print("Records:", len(df))
print("Matched conditions:", len(coverage))
print("Algorithms:", len(methods))
print("Environments:", sorted(df.environment.unique()))
print("Shifts:", sorted(df["shift"].unique()))
print("Seeds:", sorted(df.seed.unique()))

# ------------------------------------------------------------
# 2. Calculate AUC from 1000 to 5000 steps
# ------------------------------------------------------------

steps = np.array([1000, 2000, 3000, 4000, 5000])

df["auc_1000_5000"] = np.trapz(
    df[metrics].to_numpy(),
    x=steps,
    axis=1
)

df.to_csv(OUT / "six_algorithm_seed_level_metrics.csv",
          index=False)

# ------------------------------------------------------------
# 3. RQ1: Negative transfer against target-only SAC
# ------------------------------------------------------------

print("\n========== RQ1: NEGATIVE TRANSFER ==========")

baseline = df[df.method == "SAC_TARGET_ONLY"][
    keys + ["auc_1000_5000"]
].rename(columns={"auc_1000_5000": "baseline_auc"})

rq1 = df.merge(
    baseline,
    on=keys,
    validate="many_to_one"
)

rq1["auc_advantage"] = (
    rq1["auc_1000_5000"] - rq1["baseline_auc"]
)

rq1["transfer_status"] = np.select(
    [
        rq1["auc_advantage"] > 1e-9,
        rq1["auc_advantage"] < -1e-9
    ],
    ["Positive", "Negative"],
    default="Neutral"
)

rq1_summary = (
    rq1.groupby("method")
    .agg(
        mean_auc_advantage=("auc_advantage", "mean"),
        std_auc_advantage=("auc_advantage", "std"),
        positive_transfer=(
            "transfer_status",
            lambda x: 100 * (x == "Positive").mean()
        ),
        negative_transfer=(
            "transfer_status",
            lambda x: 100 * (x == "Negative").mean()
        ),
        neutral_transfer=(
            "transfer_status",
            lambda x: 100 * (x == "Neutral").mean()
        )
    )
    .reset_index()
)

rq1_summary.to_csv(
    OUT / "RQ1_negative_transfer_six_algorithms.csv",
    index=False
)

print(rq1_summary.round(3).to_string(index=False))

# ------------------------------------------------------------
# 4. RQ2: Performance across environments and shifts
# ------------------------------------------------------------

print("\n========== RQ2: TASK RELATEDNESS ==========")

rq2 = (
    df.groupby(["environment", "shift", "method"])
    .agg(
        mean_auc=("auc_1000_5000", "mean"),
        std_auc=("auc_1000_5000", "std"),
        mean_final_return=("eval_5000", "mean"),
        std_final_return=("eval_5000", "std")
    )
    .reset_index()
)

rq2.to_csv(
    OUT / "RQ2_environment_shift_performance.csv",
    index=False
)

for env in sorted(df.environment.unique()):
    print(f"\n--- {env.upper()} ---")

    sub = rq2[rq2.environment == env]

    print(
        sub.pivot(
            index="shift",
            columns="method",
            values="mean_final_return"
        ).round(2).to_string()
    )

# ------------------------------------------------------------
# 5. RQ3: Normalized versus Mean EA-NTC
# ------------------------------------------------------------

print("\n========== RQ3: EA-NTC PAIRED ADAPTATION ==========")

norm = df[df.method == "EA_NTC_NORMALIZED"]
mean = df[df.method == "EA_NTC_UNNORMALIZED"]

paired = norm[
    keys + ["auc_1000_5000", "eval_5000"]
].merge(
    mean[keys + ["auc_1000_5000", "eval_5000"]],
    on=keys,
    suffixes=("_normalized", "_mean"),
    validate="one_to_one"
)

paired["auc_difference"] = (
    paired["auc_1000_5000_normalized"] -
    paired["auc_1000_5000_mean"]
)

paired["final_return_difference"] = (
    paired["eval_5000_normalized"] -
    paired["eval_5000_mean"]
)

paired.to_csv(
    OUT / "RQ3_EA_NTC_paired_seed_comparison.csv",
    index=False
)

rq3 = (
    paired.groupby("environment")
    .agg(
        mean_auc_difference=("auc_difference", "mean"),
        std_auc_difference=("auc_difference", "std"),
        normalized_auc_wins=(
            "auc_difference",
            lambda x: 100 * (x > 0).mean()
        ),
        mean_final_difference=("final_return_difference", "mean"),
        normalized_final_wins=(
            "final_return_difference",
            lambda x: 100 * (x > 0).mean()
        )
    )
    .reset_index()
)

rq3.to_csv(
    OUT / "RQ3_EA_NTC_adaptation_summary.csv",
    index=False
)

print("\nBy environment:")
print(rq3.round(3).to_string(index=False))

# Also calculate shift-specific EA comparisons

rq3_shift = (
    paired.groupby(["environment", "shift"])
    .agg(
        mean_auc_difference=("auc_difference", "mean"),
        mean_final_difference=("final_return_difference", "mean")
    )
    .reset_index()
)

rq3_shift.to_csv(
    OUT / "RQ3_EA_NTC_adaptation_by_shift.csv",
    index=False
)

# ------------------------------------------------------------
# 6. Direct comparison: Both EA methods vs four baselines
# ------------------------------------------------------------

print("\n========== EA-NTC VS FOUR BASELINES ==========")

comparisons = []

ea_methods = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

baselines = [
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "SAC_TARGET_ONLY"
]

wide_auc = df.pivot(
    index=keys,
    columns="method",
    values="auc_1000_5000"
)

wide_final = df.pivot(
    index=keys,
    columns="method",
    values="eval_5000"
)

for ea in ea_methods:
    for baseline_method in baselines:

        auc_diff = wide_auc[ea] - wide_auc[baseline_method]
        final_diff = wide_final[ea] - wide_final[baseline_method]

        comparisons.append({
            "EA_method": ea,
            "baseline": baseline_method,
            "mean_auc_difference": auc_diff.mean(),
            "std_auc_difference": auc_diff.std(),
            "auc_win_percentage": 100 * (auc_diff > 0).mean(),
            "mean_final_return_difference": final_diff.mean(),
            "final_return_win_percentage": 100 * (final_diff > 0).mean(),
            "matched_conditions": len(auc_diff)
        })

direct = pd.DataFrame(comparisons)

direct.to_csv(
    OUT / "EA_NTC_vs_four_baselines.csv",
    index=False
)

print(direct.round(3).to_string(index=False))

# ------------------------------------------------------------
# 7. Bootstrap confidence intervals for paired AUC differences
# ------------------------------------------------------------

print("\n========== PAIRED AUC BOOTSTRAP ==========")

rng = np.random.default_rng(42)
bootstrap_rows = []

for ea in ea_methods:
    for baseline_method in baselines:

        differences = (
            wide_auc[ea] - wide_auc[baseline_method]
        ).to_numpy()

        samples = rng.choice(
            differences,
            size=(10000, len(differences)),
            replace=True
        ).mean(axis=1)

        bootstrap_rows.append({
            "EA_method": ea,
            "baseline": baseline_method,
            "mean_difference": differences.mean(),
            "CI_2.5": np.percentile(samples, 2.5),
            "CI_97.5": np.percentile(samples, 97.5),
            "matched_conditions": len(differences)
        })

bootstrap = pd.DataFrame(bootstrap_rows)

bootstrap.to_csv(
    OUT / "paired_AUC_bootstrap_confidence_intervals.csv",
    index=False
)

print(bootstrap.round(3).to_string(index=False))

# ------------------------------------------------------------
# 8. Final output
# ------------------------------------------------------------

print("\n========== ANALYSIS COMPLETED ==========")

print("RQ1:", OUT / "RQ1_negative_transfer_six_algorithms.csv")
print("RQ2:", OUT / "RQ2_environment_shift_performance.csv")
print("RQ3:", OUT / "RQ3_EA_NTC_adaptation_summary.csv")
print("EA comparisons:", OUT / "EA_NTC_vs_four_baselines.csv")
print("Bootstrap:", OUT / "paired_AUC_bootstrap_confidence_intervals.csv")

print("\nOutput directory:", OUT.resolve())

