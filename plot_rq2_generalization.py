import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# DATA: Previously reported RQ2 environment-shift results
# ------------------------------------------------------------

data = [
    ["ant", 0.1, 283.59, 1246.62, 994.10, 84.85, 1161.51, 1256.12],
    ["ant", 0.5, 18.08, 385.51, 995.02, 66.59, 461.66, 407.58],
    ["ant", 2.0, -689.04, 135.93, 994.56, 19.21, 164.45, 148.09],
    ["ant", 5.0, 5.65, 11.09, 994.92, 0.55, 2.92, 29.11],

    ["halfcheetah", 0.1, -188.00, -16.24, -0.87, -94.00, 175.92, -43.89],
    ["halfcheetah", 0.5, -181.31, 726.37, -1.48, -174.65, 1477.49, 1157.14],
    ["halfcheetah", 2.0, -182.83, 235.80, -1.55, 156.63, 497.10, 456.37],
    ["halfcheetah", 5.0, -142.60, -121.31, -1.35, -133.88, 5.87, 136.94],

    ["hopper", 0.1, 76.18, 118.26, 94.06, 72.91, 95.19, 80.56],
    ["hopper", 0.5, 76.18, 260.72, 69.89, 72.91, 132.15, 103.00],
    ["hopper", 2.0, 96.99, 201.92, 87.84, 73.64, 156.03, 225.60],
    ["hopper", 5.0, 109.10, 46.38, 46.15, 36.78, 79.16, 146.29],

    ["walker2d", 0.1, 242.57, 201.79, 77.60, 167.96, 192.87, 111.97],
    ["walker2d", 0.5, 258.60, 358.17, 122.89, 259.28, 304.14, 357.66],
    ["walker2d", 2.0, -4.93, 221.83, 127.27, 171.63, 123.36, 262.57],
    ["walker2d", 5.0, 116.25, 229.02, 98.16, 200.72, 214.16, 220.91],
]

columns = [
    "environment", "shift", "SAC Target-Only",
    "BC-SAC", "BC-PAR", "RLPD",
    "EA-NTC Normalized", "EA-NTC Mean"
]

df = pd.DataFrame(data, columns=columns)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

out = os.path.join(
    "matched_final_5000",
    "EA_NTC_RQ2_GENERALIZATION"
)

os.makedirs(out, exist_ok=True)

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

environments = ["ant", "halfcheetah", "hopper", "walker2d"]
shifts = [0.1, 0.5, 2.0, 5.0]

methods = [
    "SAC Target-Only",
    "BC-SAC",
    "BC-PAR",
    "RLPD",
    "EA-NTC Normalized",
    "EA-NTC Mean"
]

x = np.arange(len(environments))
width = 0.12

fig, axes = plt.subplots(
    2, 2,
    figsize=(17, 11),
    sharex=True
)

axes = axes.flatten()

for ax, shift in zip(axes, shifts):

    subset = df[df["shift"] == shift]

    for i, method in enumerate(methods):

        values = [
            subset.loc[
                subset["environment"] == env, method
            ].iloc[0]
            for env in environments
        ]

        offset = (i - (len(methods)-1)/2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method
        )

    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Friction Shift = {shift}",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Ant", "HalfCheetah", "Hopper", "Walker2d"]
    )

    ax.set_ylabel("Reported Performance")
    ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "RQ2: Generalization Across Environments and Friction Shifts",
    fontsize=17,
    fontweight="bold"
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005)
)

plt.tight_layout(rect=[0, 0.10, 1, 0.94])

png = os.path.join(out, "RQ2_all_algorithms_generalization.png")
pdf = os.path.join(out, "RQ2_all_algorithms_generalization.pdf")

plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")

print("\n========== RQ2 GENERALIZATION COMPLETED ==========")
print("Environments:", len(environments))
print("Friction shifts:", len(shifts))
print("Algorithms:", len(methods))
print("Conditions:", len(df))
print("\nPNG:", os.path.abspath(png))
print("PDF:", os.path.abspath(pdf))

plt.show()
