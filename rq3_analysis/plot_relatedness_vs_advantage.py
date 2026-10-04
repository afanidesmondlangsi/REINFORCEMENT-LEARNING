import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 1. PATHS
# ============================================================

input_file = (
    "rq3_analysis/"
    "rq3_effective_conditions_within_environment.csv"
)

output_dir = "rq3_analysis/figures"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

# These are the variables used in the final
# within-environment analysis.
x_col = "mean_distance_mean_z"
y_col = "bcpar_advantage_z"

data = df[
    [
        "environment",
        "shift",
        x_col,
        y_col
    ]
].dropna().copy()


# ============================================================
# 3. CORRELATIONS
# ============================================================

pearson_r, pearson_p = pearsonr(
    data[x_col],
    data[y_col]
)

spearman_rho, spearman_p = spearmanr(
    data[x_col],
    data[y_col]
)

print("\nEffective conditions:")
print(len(data))

print("\nPearson:")
print("r =", pearson_r)
print("p =", pearson_p)

print("\nSpearman:")
print("rho =", spearman_rho)
print("p =", spearman_p)


# ============================================================
# 4. SETTINGS
# ============================================================

markers = {
    "ant-friction": "o",
    "halfcheetah-friction": "s",
    "hopper-friction": "^",
    "walker2d-friction": "D"
}

display_names = {
    "ant-friction": "Ant",
    "halfcheetah-friction": "HalfCheetah",
    "hopper-friction": "Hopper",
    "walker2d-friction": "Walker2d"
}


# ============================================================
# 5. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)

for environment in markers:

    env_data = data[
        data["environment"] == environment
    ]

    ax.scatter(
        env_data[x_col],
        env_data[y_col],
        marker=markers[environment],
        s=85,
        alpha=0.8,
        label=display_names[environment]
    )

    # Label each point with its nominal shift.
    for _, row in env_data.iterrows():

        ax.annotate(
            str(row["shift"]),
            (
                row[x_col],
                row[y_col]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )


# ============================================================
# 6. REGRESSION LINE
# ============================================================

x = data[x_col].to_numpy()
y = data[y_col].to_numpy()

slope, intercept = np.polyfit(
    x,
    y,
    1
)

x_line = np.linspace(
    x.min(),
    x.max(),
    200
)

y_line = (
    slope * x_line
    + intercept
)

ax.plot(
    x_line,
    y_line,
    linestyle="--",
    linewidth=1.8,
    label="Linear trend"
)


# ============================================================
# 7. ZERO REFERENCE LINES
# ============================================================

ax.axhline(
    0,
    linewidth=1,
    linestyle=":"
)

ax.axvline(
    0,
    linewidth=1,
    linestyle=":"
)


# ============================================================
# 8. STATISTICAL ANNOTATION
# ============================================================

statistics_text = (
    f"Pearson r = {pearson_r:.3f}, "
    f"p = {pearson_p:.3f}\n"
    f"Spearman rho = {spearman_rho:.3f}, "
    f"p = {spearman_p:.3f}"
)

ax.text(
    0.03,
    0.97,
    statistics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.15
    )
)


# ============================================================
# 9. LABELS
# ============================================================

ax.set_xlabel(
    "Within-environment standardized "
    "BC-PAR dynamics distance"
)

ax.set_ylabel(
    "Within-environment standardized "
    "BC-PAR advantage over best baseline"
)

ax.set_title(
    "RQ3: Learned Dynamics Distance vs. "
    "BC-PAR Performance Advantage"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)

fig.tight_layout()


# ============================================================
# 10. SAVE
# ============================================================

png_file = os.path.join(
    output_dir,
    "rq3_distance_vs_advantage.png"
)

pdf_file = os.path.join(
    output_dir,
    "rq3_distance_vs_advantage.pdf"
)

fig.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved:")
print(png_file)
print(pdf_file)


# ============================================================
# 11. PRINT DATA USED
# ============================================================

print("\n================================================")
print("POINTS USED IN FIGURE")
print("================================================")

print(
    data.sort_values(
        ["environment", "shift"]
    ).to_string(index=False)
)

print(
    "\nHopper 0.5 is intentionally absent because "
    "Hopper 0.1 and 0.5 were experimentally verified "
    "to represent the same effective condition."
)