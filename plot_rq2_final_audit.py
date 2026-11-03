import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = r"C:\Users\LANGS\Documents\off-dynamics-rl\matched_final_5000\RQ2_FINAL_AUC_AUDIT"
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

condition = pd.read_csv(os.path.join(BASE, "RQ2_condition_level_AUC.csv"))
environment = pd.read_csv(os.path.join(BASE, "RQ2_ENVIRONMENT_SUMMARY.csv"))
frequency = pd.read_csv(os.path.join(BASE, "RQ2_FINAL_TRANSFER_FREQUENCY.csv"))
seed_frequency = pd.read_csv(os.path.join(BASE, "RQ2_SEED_LEVEL_FREQUENCY.csv"))

methods = [
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

labels = {
    "BC_SAC": "BC-SAC",
    "BC_PAR": "BC-PAR",
    "RLPD": "RLPD",
    "EA_NTC_NORMALIZED": "EA-NTC Normalized",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean",
    "SAC_TARGET_ONLY": "SAC Target-Only"
}

colors = {
    "BC_SAC": "#4472C4",
    "BC_PAR": "#70AD47",
    "RLPD": "#ED7D31",
    "EA_NTC_NORMALIZED": "#7030A0",
    "EA_NTC_UNNORMALIZED": "#C00000"
}

envs = ["ant", "halfcheetah", "hopper", "walker2d"]
shifts = [0.1, 0.5, 2.0, 5.0]

def save_figure(fig, name):
    fig.savefig(
        os.path.join(OUT, name + ".png"),
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(OUT, name + ".pdf"),
        bbox_inches="tight"
    )
    plt.close(fig)
    print("Generated:", name)

def format_millions(x, pos):
    return f"{x/1e6:.1f}"

# Consistent method ordering
for df in [condition, environment, frequency, seed_frequency]:
    df["method"] = pd.Categorical(
        df["method"], categories=methods, ordered=True
    )

# =========================================================
# FIGURE 1: CROSS-ENVIRONMENT GENERALIZATION
# =========================================================

fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(envs))
width = 0.15

for i, method in enumerate(methods):
    subset = (
        environment[environment["method"] == method]
        .set_index("environment")
        .reindex(envs)
    )

    offsets = (i - (len(methods)-1)/2) * width

    ax.bar(
        x + offsets,
        subset["mean_delta_AUC"],
        width,
        label=labels[method],
        color=colors[method]
    )

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(["Ant", "HalfCheetah", "Hopper", "Walker2d"])
ax.set_ylabel("Mean AUC advantage over SAC Target-Only")
ax.set_title("RQ2: Cross-Task Generalization Across Four Environments")
ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
ax.legend(ncol=3, fontsize=9)
ax.grid(axis="y", alpha=0.25)

save_figure(fig, "RQ2_A_cross_environment_generalization")

# =========================================================
# FIGURE 2: FRICTION-SHIFT GENERALIZATION
# Four panels, one per environment
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
axes = axes.flatten()

for ax, env in zip(axes, envs):

    subset_env = condition[condition["environment"] == env]

    for method in methods:
        data = (
            subset_env[subset_env["method"] == method]
            .sort_values("shift")
        )

        ax.plot(
            data["shift"],
            data["mean_delta_AUC"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=labels[method],
            color=colors[method]
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(env.capitalize())
    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Mean AUC advantage")
    ax.set_xticks(shifts)
    ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
    ax.grid(alpha=0.25)

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    ncol=3,
    fontsize=9
)

fig.suptitle(
    "RQ2: Generalization Across Friction-Shift Levels",
    fontsize=14
)

fig.tight_layout(rect=[0, 0.10, 1, 0.95])

save_figure(fig, "RQ2_B_friction_shift_generalization")

# =========================================================
# FIGURE 3: CONDITION-LEVEL VS SEED-LEVEL TRANSFER
# =========================================================

freq = frequency.set_index("method").reindex(methods)
seed = seed_frequency.set_index("method").reindex(methods)

condition_pct = freq["positive_pct"].fillna(0).to_numpy()

seed_pct = (
    seed["positive_seed_cases"] /
    seed["total_seed_cases"] * 100
).fillna(0).to_numpy()

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width/2,
    condition_pct,
    width,
    label="Positive condition frequency (16 conditions)",
    color="#4472C4"
)

ax.bar(
    x + width/2,
    seed_pct,
    width,
    label="Positive seed frequency (48 comparisons)",
    color="#ED7D31"
)

ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in methods], rotation=15, ha="right")
ax.set_ylabel("Positive transfer (%)")
ax.set_ylim(0, 110)
ax.set_title("RQ2: Cross-Task Transfer Consistency")
ax.legend()
ax.grid(axis="y", alpha=0.25)

for i, value in enumerate(condition_pct):
    ax.text(i-width/2, value+1.5, f"{value:.1f}%", ha="center", fontsize=8)

for i, value in enumerate(seed_pct):
    ax.text(i+width/2, value+1.5, f"{value:.1f}%", ha="center", fontsize=8)

save_figure(fig, "RQ2_C_transfer_consistency")

# =========================================================
# FIGURE 4: ENVIRONMENT-SHIFT HEATMAPS
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

vmax = condition["mean_delta_AUC"].abs().max()

for i, method in enumerate(methods):
    ax = axes[i]

    subset = condition[condition["method"] == method]

    matrix = (
        subset.pivot(
            index="environment",
            columns="shift",
            values="mean_delta_AUC"
        )
        .reindex(index=envs, columns=shifts)
    )

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_title(labels[method])
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels(shifts)
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([e.capitalize() for e in envs])

    for r in range(len(envs)):
        for c in range(len(shifts)):
            value = matrix.iloc[r, c]
            if pd.notna(value):
                ax.text(
                    c, r,
                    f"{value/1e6:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

axes[-1].axis("off")

fig.colorbar(
    image,
    ax=axes.tolist(),
    shrink=0.75,
    label="AUC advantage over SAC (millions)"
)

fig.suptitle(
    "RQ2: AUC Transfer Advantage by Environment and Friction Shift",
    fontsize=14
)

fig.tight_layout()

save_figure(fig, "RQ2_D_environment_shift_heatmaps")

print("\n========== RQ2 FIGURE GENERATION COMPLETE ==========")
print("Figures generated: 4")
print("PNG files: 4")
print("PDF files: 4")
print("Output:", OUT)
