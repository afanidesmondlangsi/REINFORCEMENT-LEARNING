import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS
# ============================================================

input_file = "rq3_analysis/rq3_diagnostics.csv"
output_dir = "rq3_analysis/figures"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_file)

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"])
df["step"] = pd.to_numeric(df["step"])
df["value"] = pd.to_numeric(df["value"])


# ============================================================
# 3. KEEP BC-PAR DISTANCE ONLY
# ============================================================

distance = df[
    (df["algorithm"] == "BC_PAR")
    & (df["metric"] == "train/distance")
].copy()


# ============================================================
# 4. CHECK DATA
# ============================================================

print("\nDistance observations:")
print(len(distance))

print("\nSteps:")
print(sorted(distance["step"].unique()))

print("\nObservations by environment and shift:")
print(
    distance.groupby(
        ["environment", "shift"]
    )
    .size()
)


# ============================================================
# 5. SUMMARIZE ACROSS SEEDS
# ============================================================

summary = (
    distance.groupby(
        [
            "environment",
            "shift",
            "step"
        ]
    )["value"]
    .agg(
        mean="mean",
        sd="std",
        n="count"
    )
    .reset_index()
)

print("\nSeeds per summarized point:")
print(
    summary["n"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. SETTINGS
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

shifts = [
    0.1,
    0.5,
    2.0,
    5.0
]

markers = {
    0.1: "o",
    0.5: "s",
    2.0: "^",
    5.0: "D"
}


# ============================================================
# 7. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
    sharex=True
)

axes = axes.flatten()


for ax, environment in zip(
    axes,
    environments
):

    env_data = summary[
        summary["environment"] == environment
    ]

    for shift in shifts:

        shift_data = env_data[
            np.isclose(
                env_data["shift"],
                shift
            )
        ].sort_values("step")

        if shift_data.empty:
            continue

        x = shift_data["step"].to_numpy()
        mean = shift_data["mean"].to_numpy()
        sd = shift_data["sd"].to_numpy()

        ax.plot(
            x,
            mean,
            marker=markers[shift],
            markersize=5,
            linewidth=2,
            label=f"Shift {shift}"
        )

        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.15
        )

    ax.set_title(
        environment_titles[environment]
    )

    ax.set_xlabel(
        "Training step"
    )

    ax.set_ylabel(
        "BC-PAR learned dynamics distance"
    )

    ax.grid(
        alpha=0.25
    )


# ============================================================
# 8. LEGEND AND TITLE
# ============================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.97)
)

fig.suptitle(
    "RQ3: BC-PAR Learned Source–Target Dynamics Distance",
    fontsize=15,
    y=1.01
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# 9. SAVE
# ============================================================

png_file = os.path.join(
    output_dir,
    "rq3_bcpar_distance.png"
)

pdf_file = os.path.join(
    output_dir,
    "rq3_bcpar_distance.pdf"
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
# 10. FINAL DISTANCE TABLE
# ============================================================

final_distance = summary[
    summary["step"] == 10000
].copy()

print("\n================================================")
print("FINAL BC-PAR DISTANCE AT STEP 10000")
print("================================================")

print(
    final_distance[
        [
            "environment",
            "shift",
            "mean",
            "sd",
            "n"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. HOPPER VALIDATION
# ============================================================

hopper = final_distance[
    final_distance["environment"]
    == "hopper-friction"
]

hopper_01 = hopper[
    np.isclose(
        hopper["shift"],
        0.1
    )
]["mean"].iloc[0]

hopper_05 = hopper[
    np.isclose(
        hopper["shift"],
        0.5
    )
]["mean"].iloc[0]

print("\n================================================")
print("HOPPER 0.1 VS 0.5 VALIDATION")
print("================================================")

print(
    "Hopper 0.1 final mean distance:",
    hopper_01
)

print(
    "Hopper 0.5 final mean distance:",
    hopper_05
)

print(
    "Difference:",
    abs(hopper_01 - hopper_05)
)