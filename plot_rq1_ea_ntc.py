import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = r"C:\Users\LANGS\Documents\off-dynamics-rl\matched_final_5000\final_three_RQ_analysis\all_learning_curve_records.csv"
OUT = r"matched_final_5000\EA_NTC_RQ1_RESULTS"

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

required = {
    "environment", "shift", "seed",
    "method", "checkpoint", "target_return", "step"
}

if not required.issubset(df.columns):
    raise ValueError("Required columns are missing.")

methods = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "SAC_TARGET_ONLY"
]

environments = ["ant", "halfcheetah", "hopper", "walker2d"]

print("Records:", len(df))
print("Methods:", df["method"].nunique())
print("Environments:", df["environment"].nunique())

# ------------------------------------------------------------
# 1. CALCULATE AUC FOR EACH MATCHED CONDITION
# ------------------------------------------------------------

records = []

keys = ["environment", "shift", "seed", "method"]

for key, group in df.groupby(keys):

    group = group.sort_values("step")

    auc = np.trapz(
        group["target_return"].values,
        group["step"].values
    )

    records.append({
        "environment": key[0],
        "shift": key[1],
        "seed": key[2],
        "method": key[3],
        "auc": auc
    })

auc_df = pd.DataFrame(records)

# Paired comparison against target-only SAC
pivot = auc_df.pivot_table(
    index=["environment", "shift", "seed"],
    columns="method",
    values="auc"
)

if pivot.isna().any().any():
    raise ValueError("Incomplete matched conditions detected.")

advantage = pivot.sub(
    pivot["SAC_TARGET_ONLY"], axis=0
)

# ------------------------------------------------------------
# 2. SUMMARY AND BOOTSTRAP CONFIDENCE INTERVALS
# ------------------------------------------------------------

rng = np.random.default_rng(42)

summary = []

for method in methods:

    values = advantage[method].values
    n = len(values)

    boot = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(10000)
    ])

    positive = np.mean(values > 0) * 100
    negative = np.mean(values < 0) * 100
    neutral = np.mean(values == 0) * 100

    summary.append({
        "method": method,
        "mean_auc_advantage": values.mean(),
        "std_auc_advantage": values.std(ddof=1),
        "positive_transfer_pct": positive,
        "negative_transfer_pct": negative,
        "neutral_transfer_pct": neutral,
        "CI_2.5": np.percentile(boot, 2.5),
        "CI_97.5": np.percentile(boot, 97.5),
        "conditions": n
    })

summary = pd.DataFrame(summary)

summary.to_csv(
    os.path.join(OUT, "RQ1_EA_NTC_statistics.csv"),
    index=False
)

advantage.to_csv(
    os.path.join(OUT, "RQ1_paired_AUC_advantages.csv")
)

print("\n========== RQ1 STATISTICS ==========")
print(summary.round(3).to_string(index=False))

# ------------------------------------------------------------
# 3. FIGURE 1: MEAN AUC ADVANTAGE
# ------------------------------------------------------------

plot_order = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "SAC_TARGET_ONLY"
]

s = summary.set_index("method").loc[plot_order]

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(s))

lower = s["mean_auc_advantage"].values - s["CI_2.5"].values
upper = s["CI_97.5"].values - s["mean_auc_advantage"].values

ax.bar(
    x,
    s["mean_auc_advantage"],
    yerr=np.vstack([lower, upper]),
    capsize=5
)

ax.axhline(0, linestyle="--", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(
    ["EA-NTC Normalized", "EA-NTC Mean",
     "BC-SAC", "BC-PAR", "RLPD", "Target-Only"],
    rotation=20
)

ax.set_ylabel("AUC advantage over target-only SAC")
ax.set_title("RQ1: EA-NTC Cumulative Transfer Advantage")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

fig.savefig(
    os.path.join(OUT, "RQ1_EA_NTC_AUC_transfer_advantage.png"),
    dpi=300
)

fig.savefig(
    os.path.join(OUT, "RQ1_EA_NTC_AUC_transfer_advantage.pdf")
)

plt.close(fig)

# ------------------------------------------------------------
# 4. FIGURE 2: POSITIVE AND NEGATIVE TRANSFER FREQUENCY
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(s))
width = 0.36

ax.bar(
    x - width/2,
    s["positive_transfer_pct"],
    width,
    label="Positive transfer"
)

ax.bar(
    x + width/2,
    s["negative_transfer_pct"],
    width,
    label="Negative transfer"
)

ax.set_xticks(x)
ax.set_xticklabels(
    ["EA-NTC Normalized", "EA-NTC Mean",
     "BC-SAC", "BC-PAR", "RLPD", "Target-Only"],
    rotation=20
)

ax.set_ylabel("Matched conditions (%)")
ax.set_title("RQ1: EA-NTC Transfer Frequency")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

fig.savefig(
    os.path.join(OUT, "RQ1_EA_NTC_transfer_frequency.png"),
    dpi=300
)

fig.savefig(
    os.path.join(OUT, "RQ1_EA_NTC_transfer_frequency.pdf")
)

plt.close(fig)

# ------------------------------------------------------------
# 5. FIGURES 3-6: LEARNING CURVES BY ENVIRONMENT
# ------------------------------------------------------------

for env in environments:

    sub = df[df["environment"] == env].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:

        method_data = sub[sub["method"] == method]

        stats = method_data.groupby("step")[
            "target_return"
        ].agg(["mean", "std"])

        steps = stats.index.values
        means = stats["mean"].values
        stds = stats["std"].fillna(0).values

        emphasis = method in [
            "EA_NTC_NORMALIZED",
            "EA_NTC_UNNORMALIZED"
        ]

        label = {
            "EA_NTC_NORMALIZED": "EA-NTC Normalized",
            "EA_NTC_UNNORMALIZED": "EA-NTC Mean",
            "BC_SAC": "BC-SAC",
            "BC_PAR": "BC-PAR",
            "RLPD": "RLPD",
            "SAC_TARGET_ONLY": "SAC Target-Only"
        }[method]

        line, = ax.plot(
            steps,
            means,
            linewidth=2.8 if emphasis else 1.5,
            label=label
        )

        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=line.get_color(),
            alpha=0.12
        )

    ax.set_xlabel("Environment interaction steps")
    ax.set_ylabel("Mean target return")
    ax.set_title(
        f"RQ1: EA-NTC Learning Curve - {env.capitalize()}"
    )

    ax.set_xticks([1000, 2000, 3000, 4000, 5000])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()

    fig.savefig(
        os.path.join(OUT, f"RQ1_EA_NTC_learning_curve_{env}.png"),
        dpi=300
    )

    fig.savefig(
        os.path.join(OUT, f"RQ1_EA_NTC_learning_curve_{env}.pdf")
    )

    plt.close(fig)

    print("Generated learning curve:", env)

print("\n========== RQ1 COMPLETED ==========")
print("Output:", os.path.abspath(OUT))
print("Figures: 6")
print("Formats: PNG and PDF")
print("Statistical summary and paired AUC data saved.")

