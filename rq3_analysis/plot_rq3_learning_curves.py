import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS
# ============================================================

input_file = "rq3_analysis/rq3_learning_curves.csv"
output_dir = "rq3_analysis/figures"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"])
df["step"] = pd.to_numeric(df["step"])


# ============================================================
# 3. IDENTIFY SCORE COLUMN
# ============================================================

possible_score_columns = [
    "normalized_score",
    "target_normalized_score",
    "value",
    "score"
]

score_column = None

for column in possible_score_columns:
    if column in df.columns:
        score_column = column
        break

if score_column is None:
    raise ValueError(
        "Could not identify normalized score column.\n"
        f"Available columns: {list(df.columns)}"
    )

print("Using score column:", score_column)


# ============================================================
# 4. SETTINGS
# ============================================================

environments = [
    "ant-friction",
    "halfcheetah-friction",
    "hopper-friction",
    "walker2d-friction"
]

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

environment_titles = {
    "ant-friction": "Ant",
    "halfcheetah-friction": "HalfCheetah",
    "hopper-friction": "Hopper",
    "walker2d-friction": "Walker2d"
}


# ============================================================
# 5. SUMMARIZE ACROSS SEEDS
# ============================================================

summary = (
    df.groupby(
        [
            "environment",
            "shift",
            "algorithm",
            "step"
        ]
    )[score_column]
    .agg(
        mean="mean",
        sd="std",
        n="count"
    )
    .reset_index()
)

print("\nNumber of summarized rows:")
print(len(summary))

print("\nSeeds per point:")
print(
    summary["n"].value_counts().sort_index()
)


# ============================================================
# 6. CREATE ONE FIGURE PER ENVIRONMENT
# ============================================================

for environment in environments:

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True
    )

    axes = axes.flatten()

    environment_data = summary[
        summary["environment"] == environment
    ]

    for ax, shift in zip(axes, shifts):

        shift_data = environment_data[
            np.isclose(
                environment_data["shift"],
                shift
            )
        ]

        for algorithm in algorithms:

            alg_data = shift_data[
                shift_data["algorithm"] == algorithm
            ].sort_values("step")

            if alg_data.empty:
                continue

            x = alg_data["step"].to_numpy()
            mean = alg_data["mean"].to_numpy()
            sd = alg_data["sd"].to_numpy()

            ax.plot(
                x,
                mean,
                linewidth=2,
                marker="o",
                markersize=4,
                label=display_names[algorithm]
            )

            ax.fill_between(
                x,
                mean - sd,
                mean + sd,
                alpha=0.15
            )

        ax.axvline(
            x=5000,
            linestyle="--",
            linewidth=1
        )

        ax.set_title(
            f"Friction shift = {shift}"
        )

        ax.grid(
            alpha=0.25
        )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    axes[2].set_xlabel(
        "Training step"
    )

    axes[3].set_xlabel(
        "Training step"
    )

    axes[0].set_ylabel(
        "Normalized target score"
    )

    axes[2].set_ylabel(
        "Normalized target score"
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97)
    )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        f"RQ3 Offline-to-Online Adaptation: "
        f"{environment_titles[environment]}",
        fontsize=14,
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    # --------------------------------------------------------
    # Save PNG and PDF
    # --------------------------------------------------------

    base_name = (
        environment
        .replace("-friction", "")
        .replace("-", "_")
    )

    png_file = os.path.join(
        output_dir,
        f"rq3_learning_curves_{base_name}.png"
    )

    pdf_file = os.path.join(
        output_dir,
        f"rq3_learning_curves_{base_name}.pdf"
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
# 7. HOPPER NOTE
# ============================================================

print("\n================================================")
print("IMPORTANT HOPPER NOTE")
print("================================================")
print(
    "Hopper shifts 0.1 and 0.5 are retained in the figure "
    "because they are nominal benchmark conditions."
)
print(
    "However, they were experimentally verified to have the "
    "same effective floor-foot friction and identical results."
)