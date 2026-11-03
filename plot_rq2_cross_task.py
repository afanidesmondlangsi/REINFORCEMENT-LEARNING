import os
import numpy as np
import matplotlib.pyplot as plt

project = r"C:\Users\LANGS\Documents\off-dynamics-rl"
# ==========================================================
# RQ2: CROSS-TASK GENERALIZATION
# ==========================================================

methods = [
    "BC-SAC",
    "BC-PAR",
    "RLPD",
    "EA-NTC Normalized",
    "EA-NTC Mean"
]

environments = ["Ant", "HalfCheetah", "Hopper", "Walker2d"]

# Mean transfer advantage by environment
environment_data = np.array([
    [540.22, 379.84,  67.21,  99.58],
    [1090.08, 172.37, -15.13, -46.64],
    [138.23, 112.21, -25.55, 46.77],
    [543.06, 712.78, 26.02, 55.51],
    [555.66, 600.33, 49.25, 85.16]
])

# Positive transfer out of 16 conditions
positive = np.array([14, 10, 9, 13, 15])

# Worst-case transfer advantage
worst_case = np.array([-62.72, -164.97, -198.74, -49.70, -130.60])

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

out = os.path.join(
    project,
    "matched_final_5000",
    "EA_NTC_RQ2_CROSS_TASK_GENERALIZATION"
)

os.makedirs(out, exist_ok=True)

# ==========================================================
# FIGURE: THREE PANELS
# ==========================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(20, 7),
    gridspec_kw={"width_ratios": [2.3, 1, 1]}
)

x = np.arange(len(environments))
width = 0.15

# Panel A: Environment-level transfer advantage
ax = axes[0]

for i, method in enumerate(methods):
    offset = (i - 2) * width

    ax.bar(
        x + offset,
        environment_data[i],
        width,
        label=method
    )

ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(environments, rotation=15)
ax.set_ylabel("Mean transfer advantage")
ax.set_title("(a) Cross-environment performance", fontweight="bold")
ax.grid(axis="y", alpha=0.25)

# Panel B: Positive transfer frequency
ax = axes[1]

bars = ax.bar(methods, positive)

ax.set_ylim(0, 17)
ax.set_ylabel("Positive conditions (out of 16)")
ax.set_title("(b) Positive transfer frequency", fontweight="bold")
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(
    ["BC-SAC", "BC-PAR", "RLPD", "EA-NTC\nNorm.", "EA-NTC\nMean"],
    rotation=20
)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, positive):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.3,
        f"{value}/16\n({value/16*100:.1f}%)",
        ha="center",
        fontsize=9
    )

# Panel C: Worst-case performance
ax = axes[2]

bars = ax.bar(methods, worst_case)

ax.axhline(0, color="black", linewidth=0.9)
ax.set_ylabel("Worst-case transfer advantage")
ax.set_title("(c) Worst-case performance", fontweight="bold")
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(
    ["BC-SAC", "BC-PAR", "RLPD", "EA-NTC\nNorm.", "EA-NTC\nMean"],
    rotation=20
)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, worst_case):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value - 8,
        f"{value:.2f}",
        ha="center",
        va="top",
        fontsize=8
    )

# ==========================================================
# FINAL FORMATTING
# ==========================================================

fig.suptitle(
    "RQ2: Cross-Task Generalization Across Four MuJoCo Environments",
    fontsize=16,
    fontweight="bold"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    bbox_to_anchor=(0.5, -0.02),
    frameon=False
)

plt.tight_layout(rect=[0, 0.12, 1, 0.92])

png = os.path.join(
    out, "RQ2_cross_task_generalization.png"
)

pdf = os.path.join(
    out, "RQ2_cross_task_generalization.pdf"
)

plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")

print("\n========== RQ2 CROSS-TASK GENERALIZATION ==========")
print("Algorithms:", len(methods))
print("Environments:", len(environments))
print("Positive-transfer conditions:", positive.sum())
print("Total algorithm conditions:", len(methods)*16)
print("\nPNG:", png)
print("PDF:", pdf)

plt.show()

