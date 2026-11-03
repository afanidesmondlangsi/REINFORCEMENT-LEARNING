import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# FINAL CONSOLIDATED ANALYSIS: RQ1, RQ2 AND RQ3
# ============================================================

ROOT = Path("matched_final_5000")
DATA = ROOT / "matched_final_288_long.csv"

PREVIOUS = ROOT / "final_three_RQ_analysis"
OLD_TABLES = PREVIOUS / "paper_tables"

OUT = ROOT / "FINAL_CONSOLIDATED_RESULTS"
TABLES = OUT / "paper_tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

METRICS = [
    "eval_1000",
    "eval_2000",
    "eval_3000",
    "eval_4000",
    "eval_5000"
]

ENVIRONMENTS = ["ant", "halfcheetah", "hopper", "walker2d"]

SHIFTS = [0.1, 0.5, 2.0, 5.0]

KEYS = ["environment", "shift", "seed"]

print("\n========== FINAL CONSOLIDATED ANALYSIS ==========")

# ------------------------------------------------------------
# 1. LOAD AND VALIDATE MAIN DATASET
# ------------------------------------------------------------

d = pd.read_csv(DATA)

required = [
    "method", "environment", "shift", "seed", "path"
] + METRICS

missing = [c for c in required if c not in d.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

d["shift"] = pd.to_numeric(d["shift"])
d["seed"] = pd.to_numeric(d["seed"])

for col in METRICS:
    d[col] = pd.to_numeric(d[col], errors="coerce")

assert len(d) == 288, f"Expected 288 records, found {len(d)}"

assert d.duplicated(["method", "environment", "shift", "seed"]).sum() == 0

assert set(d["method"].unique()) == set(METHODS)

assert set(d["environment"].unique()) == set(ENVIRONMENTS)

assert sorted(d["seed"].unique()) == [0, 1, 2]

assert d[METRICS].notna().all().all()

coverage = (
    d.groupby(["environment", "shift", "method"])
    ["seed"].nunique()
)

assert len(coverage) == 96
assert (coverage == 3).all()

print("Main records:", len(d))
print("Algorithms:", d["method"].nunique())
print("Environments:", d["environment"].nunique())
print("Matched conditions:", len(coverage) // 2)
print("Seeds:", sorted(d.seed.unique()))
print("Missing evaluation values:", d[METRICS].isna().sum().sum())

# ------------------------------------------------------------
# 2. RQ1: NEGATIVE TRANSFER
# ------------------------------------------------------------

print("\n========== RQ1: NEGATIVE TRANSFER ==========")

rq1_path = OLD_TABLES / "RQ1_negative_transfer_summary.csv"

if not rq1_path.exists():
    raise FileNotFoundError(rq1_path)

rq1 = pd.read_csv(rq1_path)

print("\nExisting RQ1 columns:")
print(rq1.columns.tolist())

rq1_required = [
    "method",
    "mean_auc_advantage",
    "positive_transfer",
    "negative_transfer"
]

if not all(c in rq1.columns for c in rq1_required):
    raise ValueError(
        "Existing RQ1 table does not have the expected columns. "
        "Inspect its schema before proceeding."
    )

rq1 = rq1[rq1["method"].isin(METHODS)].copy()

rq1["algorithm_rank_auc"] = (
    rq1["mean_auc_advantage"]
    .rank(ascending=False, method="min")
)

rq1["positive_transfer_rank"] = (
    rq1["positive_transfer"]
    .rank(ascending=False, method="min")
)

rq1.to_csv(
    TABLES / "RQ1_final_negative_transfer.csv",
    index=False
)

# Preserve original seed-level AUC calculations

seed_auc = OLD_TABLES / "RQ1_seed_level_transfer_advantage.csv"

if seed_auc.exists():
    pd.read_csv(seed_auc).to_csv(
        TABLES / "RQ1_seed_level_transfer_advantage.csv",
        index=False
    )

print("\nRQ1 final comparison:")
print(
    rq1.sort_values("mean_auc_advantage", ascending=False)
    .round(3).to_string(index=False)
)

# RQ1 figure

fig, ax = plt.subplots(figsize=(11, 6))

plot_data = rq1.sort_values(
    "mean_auc_advantage", ascending=False
)

ax.bar(
    plot_data["method"],
    plot_data["mean_auc_advantage"]
)

ax.set_title("RQ1: Mean Cumulative AUC Transfer Advantage")
ax.set_ylabel("Mean AUC advantage")
ax.set_xlabel("Algorithm")
ax.tick_params(axis="x", rotation=30)

fig.tight_layout()

fig.savefig(
    FIGURES / "RQ1_mean_AUC_transfer_advantage.png",
    dpi=300
)

plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(plot_data))
width = 0.35

ax.bar(
    x - width/2,
    plot_data["positive_transfer"],
    width,
    label="Positive transfer"
)

ax.bar(
    x + width/2,
    plot_data["negative_transfer"],
    width,
    label="Negative transfer"
)

ax.set_xticks(x)
ax.set_xticklabels(plot_data["method"], rotation=30)
ax.set_ylabel("Percentage (%)")
ax.set_title("RQ1: Positive and Negative Transfer Frequency")
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES / "RQ1_transfer_frequency.png",
    dpi=300
)

plt.close(fig)

# ------------------------------------------------------------
# 3. RQ2: PERFORMANCE BY TASK RELATEDNESS
# ------------------------------------------------------------

print("\n========== RQ2: TASK RELATEDNESS ==========")

rq2 = (
    d.groupby(["environment", "shift", "method"])[METRICS]
    .agg(["mean", "std", "count"])
)

rq2.to_csv(
    TABLES / "RQ2_all_checkpoints_mean_std.csv"
)

# Final return table

final = (
    d.groupby(["environment", "shift", "method"])
    ["eval_5000"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

final["rank"] = (
    final.groupby(["environment", "shift"])["mean"]
    .rank(ascending=False, method="min")
)

final.to_csv(
    TABLES / "RQ2_final_return_mean_std_rank.csv",
    index=False
)

print("\nRQ2 final returns:")

for env in ENVIRONMENTS:

    print(f"\n========== {env.upper()} ==========")

    sub = final[final.environment == env]

    pivot = sub.pivot(
        index="method",
        columns="shift",
        values="mean"
    )

    print(pivot.round(3).to_string())

# RQ2 learning curves: mean +/- standard deviation

for env in ENVIRONMENTS:

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:

        sub = d[
            (d.environment == env) &
            (d.method == method)
        ]

        means = sub[METRICS].mean()
        stds = sub[METRICS].std()

        ax.errorbar(
            [1000, 2000, 3000, 4000, 5000],
            means.values,
            yerr=stds.values,
            marker="o",
            capsize=3,
            label=method
        )

    ax.set_title(f"Learning Curves: {env.upper()}")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Evaluation Return")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"RQ2_learning_curves_{env}.png",
        dpi=300
    )

    plt.close(fig)

# RQ2 final performance comparison by environment

for env in ENVIRONMENTS:

    sub = final[final.environment == env]

    fig, ax = plt.subplots(figsize=(11, 6))

    methods = METHODS
    shifts = SHIFTS

    x = np.arange(len(shifts))
    width = 0.12

    for i, method in enumerate(methods):

        vals = []

        for shift in shifts:

            result = sub[
                (sub.method == method) &
                (sub.shift == shift)
            ]["mean"]

            vals.append(
                result.iloc[0] if len(result) else np.nan
            )

        ax.bar(
            x + (i - 2.5) * width,
            vals,
            width,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(shifts)

    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Evaluation return at 5,000 steps")

    ax.set_title(
        f"RQ2: Algorithm Performance Across Shifts - {env.upper()}"
    )

    ax.legend(fontsize=7)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"RQ2_shift_comparison_{env}.png",
        dpi=300
    )

    plt.close(fig)

# ------------------------------------------------------------
# 4. RQ3: NORMALIZED VS UNNORMALIZED EA-NTC
# ------------------------------------------------------------

print("\n========== RQ3: NORMALIZED VS UNNORMALIZED ==========")

norm = d[
    d.method == "EA_NTC_NORMALIZED"
].copy()

unnorm = d[
    d.method == "EA_NTC_UNNORMALIZED"
].copy()

paired = norm[
    KEYS + METRICS
].merge(
    unnorm[KEYS + METRICS],
    on=KEYS,
    suffixes=("_normalized", "_unnormalized"),
    validate="one_to_one"
)

assert len(paired) == 48

difference_rows = []

for metric in METRICS:

    col = metric + "_difference"

    paired[col] = (
        paired[metric + "_normalized"] -
        paired[metric + "_unnormalized"]
    )

    for _, row in paired.iterrows():

        difference_rows.append({
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "checkpoint": metric,
            "difference": row[col],
            "winner": (
                "NORMALIZED"
                if row[col] > 0
                else "UNNORMALIZED"
                if row[col] < 0
                else "TIE"
            )
        })

paired.to_csv(
    TABLES / "RQ3_paired_seed_comparisons.csv",
    index=False
)

diff_df = pd.DataFrame(difference_rows)

diff_df.to_csv(
    TABLES / "RQ3_paired_checkpoint_differences_long.csv",
    index=False
)

# Paired differences summarized by environment and shift

paired_summary = (
    diff_df.groupby(
        ["environment", "shift", "checkpoint"]
    )
    .agg(
        mean_difference=("difference", "mean"),
        std_difference=("difference", "std"),
        normalized_wins=(
            "winner",
            lambda x: (x == "NORMALIZED").sum()
        ),
        unnormalized_wins=(
            "winner",
            lambda x: (x == "UNNORMALIZED").sum()
        ),
        ties=("winner", lambda x: (x == "TIE").sum()),
        seeds=("difference", "count")
    )
    .reset_index()
)

paired_summary.to_csv(
    TABLES / "RQ3_paired_summary_by_shift.csv",
    index=False
)

# Overall environment comparison

overall = (
    diff_df.groupby(
        ["environment", "checkpoint"]
    )
    .agg(
        mean_difference=("difference", "mean"),
        std_difference=("difference", "std"),
        normalized_wins=(
            "winner",
            lambda x: (x == "NORMALIZED").sum()
        ),
        unnormalized_wins=(
            "winner",
            lambda x: (x == "UNNORMALIZED").sum()
        )
    )
    .reset_index()
)

overall.to_csv(
    TABLES / "RQ3_overall_environment_comparison.csv",
    index=False
)

print("\nRQ3 final return differences:")

print(
    paired_summary[
        paired_summary.checkpoint == "eval_5000"
    ].round(4).to_string(index=False)
)

# RQ3 paired difference figures

for env in ENVIRONMENTS:

    fig, ax = plt.subplots(figsize=(9, 6))

    sub = paired_summary[
        (paired_summary.environment == env) &
        (paired_summary.checkpoint == "eval_5000")
    ]

    ax.errorbar(
        sub["shift"],
        sub["mean_difference"],
        yerr=sub["std_difference"],
        marker="o",
        capsize=4
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(
        f"RQ3: Normalized - Unnormalized EA-NTC ({env.upper()})"
    )

    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Final return difference")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"RQ3_normalized_vs_unnormalized_{env}.png",
        dpi=300
    )

    plt.close(fig)

# ------------------------------------------------------------
# 5. CROSS-ENVIRONMENT ALGORITHM RANKINGS
# ------------------------------------------------------------

print("\n========== CROSS-ENVIRONMENT RANKINGS ==========")

rankings = (
    final.groupby("method")
    .agg(
        mean_final_return=("mean", "mean"),
        mean_rank=("rank", "mean"),
        conditions_won=("rank", lambda x: (x == 1).sum()),
        conditions=("rank", "count")
    )
    .reset_index()
    .sort_values("mean_rank")
)

rankings.to_csv(
    TABLES / "cross_environment_algorithm_rankings.csv",
    index=False
)

print(rankings.round(3).to_string(index=False))

# ------------------------------------------------------------
# 6. GENERATE CONSOLIDATED REPORT
# ------------------------------------------------------------

report = []

report.append("# Final Consolidated Experimental Results")
report.append("")
report.append("## Experimental design")
report.append("")
report.append("- Four environments: Ant, HalfCheetah, Hopper and Walker2d.")
report.append("- Four friction shifts: 0.1, 0.5, 2.0 and 5.0.")
report.append("- Three seeds: 0, 1 and 2.")
report.append("- Six algorithms.")
report.append("- 288 total experimental records.")
report.append("- Evaluation checkpoints: 1,000 through 5,000 steps.")
report.append("")
report.append("## RQ1: Negative transfer")
report.append("")
report.append(rq1.to_markdown(index=False))
report.append("")
report.append("## RQ2: Task relatedness")
report.append("")
report.append(final.to_markdown(index=False))
report.append("")
report.append("## RQ3: Normalized versus unnormalized EA-NTC")
report.append("")
report.append(overall.to_markdown(index=False))
report.append("")
report.append("## Cross-environment rankings")
report.append("")
report.append(rankings.to_markdown(index=False))
report.append("")
report.append("## Statistical interpretation")
report.append("")
report.append(
    "All reported means and standard deviations are calculated "
    "from the three available seeds. The paired EA-NTC comparisons "
    "use identical environment, shift and seed conditions. "
    "No statistical-significance claims are made from these "
    "descriptive comparisons alone."
)
report.append("")
report.append(
    "AUC transfer advantages in RQ1 are retained from the existing "
    "RQ1 analysis to preserve the original integration convention."
)

report_path = OUT / "FINAL_CONSOLIDATED_RESEARCH_REPORT.md"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 7. FINAL OUTPUT
# ------------------------------------------------------------

print("\n========== FINAL ANALYSIS COMPLETED ==========")

print("Main dataset:", len(d))
print("RQ1 methods:", len(rq1))
print("RQ2 environment/shift/method records:", len(final))
print("RQ3 paired records:", len(paired))
print("RQ3 checkpoint differences:", len(diff_df))

print("\nOUTPUT DIRECTORY:")
print(OUT.resolve())

print("\nPAPER TABLES:")
print(TABLES.resolve())

print("\nFIGURES:")
print(FIGURES.resolve())

print("\nFINAL REPORT:")
print(report_path.resolve())

