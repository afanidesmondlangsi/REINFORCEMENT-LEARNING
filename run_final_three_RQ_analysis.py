import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("matched_final_5000")
DATA = ROOT / "matched_final_288_long.csv"
DIAG = ROOT / "ea_ntc_diagnostics" / "ea_ntc_diagnostics_timeseries.csv"

OUT = ROOT / "final_three_RQ_analysis"
FIG = OUT / "figures"
TABLES = OUT / "paper_tables"

for folder in [OUT, FIG, TABLES]:
    folder.mkdir(parents=True, exist_ok=True)

ENVIRONMENTS = ["hopper", "halfcheetah", "walker2d", "ant"]
SHIFTS = [0.1, 0.5, 2.0, 5.0]
SEEDS = [0, 1, 2]

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_UNNORMALIZED",
    "EA_NTC_NORMALIZED"
]

EA_METHODS = [
    "EA_NTC_UNNORMALIZED",
    "EA_NTC_NORMALIZED"
]

METHOD_LABELS = {
    "SAC_TARGET_ONLY": "SAC Target Only",
    "BC_SAC": "BC-SAC",
    "RLPD": "RLPD",
    "BC_PAR": "BC-PAR",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean",
    "EA_NTC_NORMALIZED": "EA-NTC Normalized"
}

STEPS = [1000, 2000, 3000, 4000, 5000]

EVAL_COLS = {
    1000: "eval_1000",
    2000: "eval_2000",
    3000: "eval_3000",
    4000: "eval_4000",
    5000: "eval_5000"
}

KEYS = ["environment", "shift", "seed"]

print("\n========== FINAL THREE-RQ ANALYSIS ==========")

# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(DATA)

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)

assert len(df) == 288, f"Expected 288 records, got {len(df)}"

assert set(df["environment"]) == set(ENVIRONMENTS)
assert set(df["method"]) == set(METHODS)
assert set(df["shift"]) == set(SHIFTS)
assert set(df["seed"]) == set(SEEDS)

assert df.duplicated(KEYS + ["method"]).sum() == 0

assert df["tag"].eq("test/target_return").all()

assert df[list(EVAL_COLS.values())].notna().all().all()

print("Main dataset:", df.shape)
print("Environments:", len(ENVIRONMENTS))
print("Methods:", len(METHODS))
print("Matched conditions:", df[KEYS].drop_duplicates().shape[0])
print("Seeds:", SEEDS)

# ============================================================
# 2. CONVERT EVALUATIONS INTO LONG FORMAT
# ============================================================

long = df.melt(
    id_vars=KEYS + ["method"],
    value_vars=list(EVAL_COLS.values()),
    var_name="checkpoint",
    value_name="target_return"
)

step_lookup = {
    value: step for step, value in EVAL_COLS.items()
}

long["step"] = long["checkpoint"].map(
    lambda x: int(x.replace("eval_", ""))
)

long = long.sort_values(
    KEYS + ["method", "step"]
)

long.to_csv(
    OUT / "all_learning_curve_records.csv",
    index=False
)

# ============================================================
# 3. CALCULATE AUC AND LEARNING METRICS
# ============================================================

records = []

for (env, shift, seed, method), group in long.groupby(
    KEYS + ["method"]
):

    group = group.sort_values("step")

    x = group["step"].to_numpy(dtype=float)
    y = group["target_return"].to_numpy(dtype=float)

    auc = np.trapz(y, x)

    records.append({
        "environment": env,
        "shift": shift,
        "seed": seed,
        "method": method,

        "return_1000": y[0],
        "return_2000": y[1],
        "return_3000": y[2],
        "return_4000": y[3],
        "return_5000": y[4],

        "auc_1000_5000": auc,
        "mean_return_auc": auc / (x[-1] - x[0]),

        "improvement_1000_5000": y[-1] - y[0],

        "early_improvement_1000_3000": y[2] - y[0],

        "late_improvement_3000_5000": y[-1] - y[2]
    })

metrics = pd.DataFrame(records)

metrics.to_csv(
    TABLES / "seed_level_performance_metrics.csv",
    index=False
)

print("\nPerformance metrics calculated:", len(metrics))

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

summary_metrics = [
    "return_1000",
    "return_2000",
    "return_3000",
    "return_4000",
    "return_5000",
    "auc_1000_5000",
    "mean_return_auc",
    "improvement_1000_5000"
]

summary = (
    metrics.groupby(
        ["environment", "shift", "method"]
    )[summary_metrics]
    .agg(["mean", "std", "count"])
)

summary.to_csv(
    TABLES / "performance_summary_mean_std.csv"
)

# ============================================================
# RQ1: NEGATIVE TRANSFER
# ============================================================

print("\n========== RQ1: NEGATIVE TRANSFER ==========")

baseline = metrics[
    metrics["method"] == "SAC_TARGET_ONLY"
][KEYS + [
    "auc_1000_5000",
    "return_5000"
]].rename(columns={
    "auc_1000_5000": "baseline_auc",
    "return_5000": "baseline_return"
})

transfer = metrics[
    metrics["method"] != "SAC_TARGET_ONLY"
].merge(
    baseline,
    on=KEYS,
    how="inner",
    validate="many_to_one"
)

transfer["transfer_auc_advantage"] = (
    transfer["auc_1000_5000"] -
    transfer["baseline_auc"]
)

transfer["transfer_return_advantage"] = (
    transfer["return_5000"] -
    transfer["baseline_return"]
)

transfer["transfer_category"] = np.select(
    [
        transfer["transfer_auc_advantage"] > 0,
        transfer["transfer_auc_advantage"] < 0
    ],
    [
        "Positive transfer",
        "Negative transfer"
    ],
    default="Neutral transfer"
)

transfer.to_csv(
    TABLES / "RQ1_seed_level_transfer_advantage.csv",
    index=False
)

rq1 = (
    transfer.groupby(
        ["environment", "shift", "method"]
    )
    .agg(
        mean_auc_advantage=(
            "transfer_auc_advantage", "mean"
        ),
        std_auc_advantage=(
            "transfer_auc_advantage", "std"
        ),
        mean_return_advantage=(
            "transfer_return_advantage", "mean"
        ),
        positive_transfer=(
            "transfer_auc_advantage",
            lambda x: (x > 0).mean() * 100
        ),
        negative_transfer=(
            "transfer_auc_advantage",
            lambda x: (x < 0).mean() * 100
        ),
        paired_conditions=(
            "transfer_auc_advantage", "count"
        )
    )
    .reset_index()
)

rq1.to_csv(
    TABLES / "RQ1_negative_transfer_summary.csv",
    index=False
)

print("\nTransfer advantage by algorithm:")
print(
    rq1.groupby("method")[
        ["mean_auc_advantage", "positive_transfer",
         "negative_transfer"]
    ].mean().round(3).to_string()
)

# ============================================================
# RQ2: TASK RELATEDNESS
# ============================================================

print("\n========== RQ2: TASK RELATEDNESS ==========")

rq2 = (
    metrics.groupby(
        ["environment", "shift", "method"]
    )
    .agg(
        mean_auc=("auc_1000_5000", "mean"),
        std_auc=("auc_1000_5000", "std"),
        mean_return_5000=("return_5000", "mean"),
        std_return_5000=("return_5000", "std"),
        mean_learning_improvement=(
            "improvement_1000_5000", "mean"
        )
    )
    .reset_index()
)

rq2.to_csv(
    TABLES / "RQ2_task_relatedness_summary.csv",
    index=False
)

# EA-NTC advantage relative to competing algorithms
comparison_methods = [
    "BC_SAC", "BC_PAR", "RLPD", "SAC_TARGET_ONLY"
]

ea_comparisons = []

for ea in EA_METHODS:
    for other in comparison_methods:

        a = metrics[
            metrics["method"] == ea
        ][KEYS + ["auc_1000_5000"]]

        b = metrics[
            metrics["method"] == other
        ][KEYS + ["auc_1000_5000"]]

        paired = a.merge(
            b,
            on=KEYS,
            suffixes=("_ea", "_comparison"),
            validate="one_to_one"
        )

        paired["ea_method"] = ea
        paired["comparison_method"] = other

        paired["auc_difference"] = (
            paired["auc_1000_5000_ea"] -
            paired["auc_1000_5000_comparison"]
        )

        ea_comparisons.append(paired)

ea_comparisons = pd.concat(
    ea_comparisons,
    ignore_index=True
)

ea_comparisons.to_csv(
    TABLES / "RQ2_EA_NTC_pairwise_comparisons.csv",
    index=False
)

# ============================================================
# RQ3: ADAPTATION
# ============================================================

print("\n========== RQ3: ADAPTATION ==========")

rq3 = (
    metrics.groupby(
        ["environment", "shift", "method"]
    )
    .agg(
        early_improvement=(
            "early_improvement_1000_3000", "mean"
        ),
        late_improvement=(
            "late_improvement_3000_5000", "mean"
        ),
        total_improvement=(
            "improvement_1000_5000", "mean"
        ),
        mean_auc=("auc_1000_5000", "mean"),
        auc_stability=("auc_1000_5000", "std")
    )
    .reset_index()
)

rq3.to_csv(
    TABLES / "RQ3_adaptation_summary.csv",
    index=False
)

# Compare normalized and mean EA-NTC directly
norm = metrics[
    metrics["method"] == "EA_NTC_NORMALIZED"
][KEYS + [
    "auc_1000_5000",
    "return_5000",
    "improvement_1000_5000"
]]

mean = metrics[
    metrics["method"] == "EA_NTC_UNNORMALIZED"
][KEYS + [
    "auc_1000_5000",
    "return_5000",
    "improvement_1000_5000"
]]

ea_pair = norm.merge(
    mean,
    on=KEYS,
    suffixes=("_normalized", "_mean"),
    validate="one_to_one"
)

for metric in [
    "auc_1000_5000",
    "return_5000",
    "improvement_1000_5000"
]:
    ea_pair[f"{metric}_difference"] = (
        ea_pair[f"{metric}_normalized"] -
        ea_pair[f"{metric}_mean"]
    )

ea_pair.to_csv(
    TABLES / "RQ3_EA_NTC_normalized_vs_mean_paired.csv",
    index=False
)

print("\nNormalized versus mean:")
print(
    ea_pair.groupby("environment")[
        [
            "auc_1000_5000_difference",
            "return_5000_difference"
        ]
    ].mean().round(3).to_string()
)

# ============================================================
# 5. EA-NTC DIAGNOSTIC ANALYSIS
# ============================================================

print("\n========== EA-NTC DIAGNOSTICS ==========")

diag = pd.read_csv(DIAG)

diag["shift"] = pd.to_numeric(diag["shift"])
diag["seed"] = pd.to_numeric(
    diag["seed"], errors="raise"
).astype(int)

diag = diag[
    diag["step"].between(0, 5000)
].copy()

diagnostic_tags = [
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/mean_source_weight",
    "ea_ntc/source_weight_ess",
    "ea_ntc/source_weight_below_0_5",
    "ea_ntc/gate_ready",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/heldout_target_mse",
    "ea_ntc/heldout_zero_delta_mse",
    "ea_ntc/dynamics_updates",
    "ea_ntc/target_dynamics_train_count",
    "ea_ntc/target_calibration_count"
]

diag = diag[
    diag["tag"].isin(diagnostic_tags)
].copy()

# Last observed value per run and diagnostic
last_diag = (
    diag.sort_values("step")
    .groupby(
        ["method", "environment", "shift",
         "seed", "tag"],
        as_index=False
    )
    .tail(1)
)

last_diag.to_csv(
    TABLES / "EA_NTC_final_run_diagnostics.csv",
    index=False
)

diag_summary = (
    last_diag.groupby(
        ["method", "environment", "shift", "tag"]
    )["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

diag_summary.to_csv(
    TABLES / "EA_NTC_diagnostic_summary.csv",
    index=False
)

# Full diagnostic progression
diag_progression = (
    diag.groupby(
        ["method", "environment", "shift", "tag", "step"]
    )["value"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

diag_progression.to_csv(
    TABLES / "EA_NTC_diagnostic_progression.csv",
    index=False
)

# ============================================================
# 6. FIGURE 1: LEARNING CURVES
# ============================================================

print("\nGenerating learning curves...")

for env in ENVIRONMENTS:

    fig, ax = plt.subplots(figsize=(10, 6))

    subset = long[long["environment"] == env]

    for method in METHODS:

        part = subset[subset["method"] == method]

        stats = (
            part.groupby("step")["target_return"]
            .agg(["mean", "std"])
        )

        ax.plot(
            stats.index,
            stats["mean"],
            marker="o",
            label=METHOD_LABELS[method]
        )

        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.10
        )

    ax.set_title(f"{env}: Target Learning Curves")
    ax.set_xlabel("Environment Interaction Steps")
    ax.set_ylabel("Target Raw Return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG / f"learning_curves_{env}.png",
        dpi=300
    )
    plt.close(fig)

# ============================================================
# 7. FIGURE 2: TRANSFER ADVANTAGE
# ============================================================

print("\nGenerating transfer advantage figures...")

for method in METHODS:
    if method == "SAC_TARGET_ONLY":
        continue

    part = transfer[
        transfer["method"] == method
    ]

    heat = (
        part.groupby(
            ["environment", "shift"]
        )["transfer_auc_advantage"]
        .mean()
        .unstack()
        .reindex(
            index=ENVIRONMENTS,
            columns=SHIFTS
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(
        heat.values,
        aspect="auto",
        cmap="RdYlGn"
    )

    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels(SHIFTS)
    ax.set_yticks(range(len(ENVIRONMENTS)))
    ax.set_yticklabels(ENVIRONMENTS)

    ax.set_xlabel("Friction Shift")
    ax.set_ylabel("Environment")
    ax.set_title(
        f"{METHOD_LABELS[method]}: "
        "Paired AUC Advantage over SAC Target Only"
    )

    for i in range(len(ENVIRONMENTS)):
        for j in range(len(SHIFTS)):
            value = heat.iloc[i, j]
            ax.text(
                j, i, f"{value:.1f}",
                ha="center", va="center"
            )

    fig.colorbar(
        im, ax=ax,
        label="Raw-return AUC advantage"
    )

    fig.tight_layout()
    fig.savefig(
        FIG / f"transfer_advantage_{method}.png",
        dpi=300
    )
    plt.close(fig)

# ============================================================
# 8. FIGURE 3: NORMALIZED VS MEAN
# ============================================================

print("\nGenerating EA-NTC comparison figures...")

for env in ENVIRONMENTS:

    part = ea_pair[
        ea_pair["environment"] == env
    ]

    grouped = (
        part.groupby("shift")[
            "auc_1000_5000_difference"
        ]
        .mean()
        .reindex(SHIFTS)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [str(x) for x in SHIFTS],
        grouped.values
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title(
        f"{env}: EA-NTC Normalized minus Mean AUC"
    )
    ax.set_xlabel("Friction Shift")
    ax.set_ylabel("Paired Raw-return AUC Difference")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG / f"EA_NTC_normalized_vs_mean_{env}.png",
        dpi=300
    )
    plt.close(fig)

# ============================================================
# 9. FIGURE 4: ADAPTIVE GATE PROGRESSION
# ============================================================

print("\nGenerating EA-NTC diagnostic figures...")

plot_tags = [
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/mean_source_weight",
    "ea_ntc/source_weight_ess",
    "ea_ntc/heldout_relative_mse"
]

for tag in plot_tags:

    part = diag_progression[
        diag_progression["tag"] == tag
    ]

    if part.empty:
        continue

    for env in ENVIRONMENTS:

        fig, ax = plt.subplots(figsize=(9, 5))

        env_data = part[
            part["environment"] == env
        ]

        for method in EA_METHODS:

            m = env_data[
                env_data["method"] == method
            ]

            if m.empty:
                continue

            grouped = m.groupby("step").agg(
                mean=("mean", "mean"),
                std=("mean", "std")
            )

            ax.plot(
                grouped.index,
                grouped["mean"],
                marker="o",
                label=METHOD_LABELS[method]
            )

        ax.set_title(
            f"{env}: {tag.split('/')[-1]}"
        )
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(tag.split("/")[-1])
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        safe_tag = tag.split("/")[-1]

        fig.savefig(
            FIG / f"{safe_tag}_{env}.png",
            dpi=300
        )

        plt.close(fig)

# ============================================================
# 10. AUTOMATIC SUMMARY REPORT
# ============================================================

report = []

report.append("# Final 5,000-Step Reinforcement Learning Analysis")
report.append("")
report.append("## Experimental design")
report.append("")
report.append("- Four environments.")
report.append("- Four friction shifts.")
report.append("- Three matched seeds: 0, 1 and 2.")
report.append("- Six algorithms.")
report.append("- Five evaluation checkpoints.")
report.append("- 288 experiment records.")
report.append("")
report.append("Primary outcome: test/target_return.")
report.append("")
report.append("AUC is calculated using trapezoidal integration "
              "from 1,000 to 5,000 steps.")
report.append("")
report.append("## RQ1: Negative transfer")
report.append("")

for method in METHODS:
    if method == "SAC_TARGET_ONLY":
        continue

    p = transfer[transfer["method"] == method]

    report.append(
        f"### {METHOD_LABELS[method]}"
    )

    report.append(
        f"- Mean paired AUC advantage: "
        f"{p['transfer_auc_advantage'].mean():.3f}"
    )

    report.append(
        f"- Positive transfer: "
        f"{(p['transfer_auc_advantage'] > 0).mean()*100:.1f}%"
    )

    report.append(
        f"- Negative transfer: "
        f"{(p['transfer_auc_advantage'] < 0).mean()*100:.1f}%"
    )

report.append("")
report.append("## RQ2: Task relatedness")
report.append("")
report.append(
    "Inspect RQ2_task_relatedness_summary.csv and "
    "RQ2_EA_NTC_pairwise_comparisons.csv."
)
report.append("")
report.append("Friction shift is treated as the experimental "
              "mismatch parameter, not as a direct measurement "
              "of source-target dynamics distance.")
report.append("")
report.append("## RQ3: Adaptation")
report.append("")
report.append(
    "Inspect RQ3_adaptation_summary.csv, "
    "RQ3_EA_NTC_normalized_vs_mean_paired.csv "
    "and the EA-NTC diagnostic outputs."
)
report.append("")
report.append("## Important limitations")
report.append("")
report.append(
    "- The six-method primary comparison uses raw target returns."
)
report.append(
    "- Normalized-score observations are not available for all "
    "six methods in the matched main dataset."
)
report.append(
    "- EA adaptive controller and transfer-quality controller "
    "were disabled in the recorded configurations."
)
report.append(
    "- Three seeds provide limited precision for statistical "
    "generalization."
)
report.append(
    "- Positive AUC advantage indicates better cumulative "
    "target return over the measured interval, not proof of "
    "causal transfer benefit."
)

(OUT / "final_research_report.md").write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print("\n========== FINAL ANALYSIS SUMMARY ==========")

print("Main records:", len(df))
print("Seed-level metrics:", len(metrics))
print("Paired transfer comparisons:", len(transfer))
print("EA normalized-mean comparisons:", len(ea_pair))
print("Diagnostic observations:", len(diag))

print("\nRQ1 table:")
print(rq1.groupby("method")[
    ["mean_auc_advantage", "positive_transfer",
     "negative_transfer"]
].mean().round(3).to_string())

print("\nRQ3 EA-NTC paired comparison:")
print(ea_pair.groupby("environment")[
    ["auc_1000_5000_difference",
     "return_5000_difference"]
].mean().round(3).to_string())

print("\n========== COMPLETED ==========")
print("Output directory:", OUT.resolve())
print("Tables:", TABLES.resolve())
print("Figures:", FIG.resolve())
print("Report:", (OUT / "final_research_report.md").resolve())
