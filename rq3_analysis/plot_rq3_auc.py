import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# ============================================================
# 1. PATHS
# ============================================================

input_file = "rq3_analysis/rq3_auc_by_run.csv"
output_dir = "rq3_analysis/figures"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"])

metric = "normalized_auc_5000_10000"


# ============================================================
# 3. SETTINGS
# ============================================================

environments = [
    "ant-friction",
    "halfcheetah-friction",
    "hopper-friction",
    "walker2d-friction"
]

environment_titles = {
    "ant-friction": "Ant",
    "halfcheetah-friction": "HalfCheetah",
    "hopper-friction": "Hopper",
    "walker2d-friction": "Walker2d"
}

shifts = [0.1, 0.5, 2.0, 5.0]

algorithms = [
    "BC_SAC",
    "RLPD",
    "BC_PAR"
]

display_names = {
    "BC_SAC": "BC-SAC",
    "RLPD": "RLPD",
    "BC_PAR": "BC-PAR"
}

markers = {
    "BC_SAC": "o",
    "RLPD": "s",
    "BC_PAR": "^"
}


# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================

summary = (
    df.groupby(
        [
            "environment",
            "shift",
            "algorithm"
        ]
    )[metric]
    .agg(
        mean="mean",
        sd="std",
        n="count"
    )
    .reset_index()
)

summary["se"] = (
    summary["sd"] /
    np.sqrt(summary["n"])
)

summary["t_critical"] = summary["n"].apply(
    lambda n: t.ppf(
        0.975,
        df=n - 1
    )
)

summary["ci_half_width"] = (
    summary["t_critical"]
    * summary["se"]
)


# ============================================================
# 5. CHECK COMPLETENESS
# ============================================================

print("\nAlgorithm-condition summaries:")
print(len(summary))

print("\nSeeds per condition:")
print(
    summary["n"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

axes = axes.flatten()

# Slight horizontal offsets so algorithms do not overlap
offsets = {
    "BC_SAC": -0.18,
    "RLPD": 0.00,
    "BC_PAR": 0.18
}

x_positions = np.arange(
    len(shifts)
)


for ax, environment in zip(
    axes,
    environments
):

    env_summary = summary[
        summary["environment"] == environment
    ]

    env_runs = df[
        df["environment"] == environment
    ]

    for algorithm in algorithms:

        alg_summary = env_summary[
            env_summary["algorithm"] == algorithm
        ].sort_values("shift")

        means = []
        errors = []

        for shift in shifts:

            row = alg_summary[
                np.isclose(
                    alg_summary["shift"],
                    shift
                )
            ]

            means.append(
                row["mean"].iloc[0]
            )

            errors.append(
                row["ci_half_width"].iloc[0]
            )

        positions = (
            x_positions
            + offsets[algorithm]
        )

        # --------------------------------------------
        # Mean + 95% CI
        # --------------------------------------------

        ax.errorbar(
            positions,
            means,
            yerr=errors,
            marker=markers[algorithm],
            markersize=7,
            linewidth=1.8,
            capsize=4,
            label=display_names[algorithm]
        )

        # --------------------------------------------
        # Individual seed values
        # --------------------------------------------

        for i, shift in enumerate(shifts):

            seed_values = env_runs[
                (env_runs["algorithm"] == algorithm)
                & np.isclose(
                    env_runs["shift"],
                    shift
                )
            ][metric].to_numpy()

            # Small deterministic spread so the three
            # seed points remain visible.
            jitter = np.linspace(
                -0.025,
                0.025,
                len(seed_values)
            )

            ax.scatter(
                positions[i] + jitter,
                seed_values,
                s=22,
                alpha=0.65
            )

    ax.set_title(
        environment_titles[environment]
    )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [str(s) for s in shifts]
    )

    ax.set_xlabel(
        "Friction shift"
    )

    ax.set_ylabel(
        "Normalized AUC (steps 5000–10000)"
    )

    ax.grid(
        alpha=0.25
    )


# ============================================================
# 7. LEGEND AND TITLE
# ============================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.97)
)

fig.suptitle(
    "RQ3 Post-Adaptation Performance",
    fontsize=15,
    y=1.01
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# 8. SAVE
# ============================================================

png_file = os.path.join(
    output_dir,
    "rq3_normalized_auc.png"
)

pdf_file = os.path.join(
    output_dir,
    "rq3_normalized_auc.pdf"
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
# 9. WIN COUNTS
# ============================================================

winners = (
    summary.loc[
        summary.groupby(
            ["environment", "shift"]
        )["mean"].idxmax()
    ]
)

print("\n================================================")
print("AUC WINNERS")
print("================================================")

print(
    winners[
        [
            "environment",
            "shift",
            "algorithm",
            "mean"
        ]
    ].to_string(index=False)
)

print("\nWinner counts:")
print(
    winners["algorithm"]
    .value_counts()
)


print("\n================================================")
print("HOPPER NOTE")
print("================================================")

print(
    "Hopper 0.1 and 0.5 are both shown as nominal benchmark "
    "conditions, although they were experimentally verified "
    "to have identical effective floor-foot friction."
)