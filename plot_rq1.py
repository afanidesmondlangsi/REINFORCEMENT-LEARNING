import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# RQ1 RESULTS
# Hopper-friction, shift = 0.1
# Seeds = 0, 1, 2
# ============================================================

interactions = np.array([0, 100, 200, 300, 400, 500])

results = {
    "Target-only SAC": {
        0: [26.3360, 26.6798, 150.3314, 76.0675, 68.9321, 126.2788],
        1: [20.6006, 20.6365, 69.3639, 54.7659, 164.6168, 56.2841],
        2: [14.2691, 14.3111, 4.5948, 45.8442, 58.1246, 45.9727]
    },

    "BC-SAC": {
        0: [26.3360, 26.6798, 139.7879, 91.0532, 61.4719, 58.7532],
        1: [20.6006, 20.6365, 141.7898, 63.2706, 109.4266, 657.6992],
        2: [14.2691, 14.3111, 69.9655, 37.3880, 88.2351, 65.7203]
    },

    "RLPD": {
        0: [17.2913, 17.6584, 45.7935, 69.8461, 67.0994, 80.7631],
        1: [210.4238, 225.1484, 50.3525, 36.8799, 105.7455, 52.2291],
        2: [16.1719, 16.1671, 81.4145, 46.4885, 75.9030, 85.7329]
    }
}


# ============================================================
# CREATE DATASET
# ============================================================

rows = []

for algorithm, seeds in results.items():

    for seed, returns in seeds.items():

        initial_return = returns[0]

        for interaction, target_return in zip(interactions, returns):

            improvement = target_return - initial_return

            rows.append([
                algorithm,
                seed,
                interaction,
                target_return,
                initial_return,
                improvement
            ])


df = pd.DataFrame(
    rows,
    columns=[
        "Algorithm",
        "Seed",
        "Target_Interactions",
        "Target_Return",
        "Initial_Return",
        "Improvement_From_Initial"
    ]
)


# Save raw dataset
df.to_csv(
    "rq1_results.csv",
    index=False
)

print("\nDataset created:")
print(df.head())

print("\nNumber of observations:", len(df))


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = (
    df.groupby(
        ["Algorithm", "Target_Interactions"]
    )["Target_Return"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\nSummary:")
print(summary)

summary.to_csv(
    "rq1_summary.csv",
    index=False
)


# ============================================================
# GRAPH 1
# MEAN TARGET RETURN +/- STANDARD DEVIATION
# ============================================================

plt.figure(figsize=(9, 6))

for algorithm in results.keys():

    temp = summary[
        summary["Algorithm"] == algorithm
    ]

    x = temp["Target_Interactions"].to_numpy()
    mean = temp["mean"].to_numpy()
    std = temp["std"].to_numpy()

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=algorithm
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.15
    )


plt.xlabel("Target Environment Interactions")
plt.ylabel("Target Evaluation Return")

plt.title(
    "RQ1: Target Performance vs Target Interactions"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "rq1_mean_learning_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2
# INDIVIDUAL SEED CURVES
# ============================================================

plt.figure(figsize=(10, 7))

for algorithm, seeds in results.items():

    for seed, returns in seeds.items():

        plt.plot(
            interactions,
            returns,
            marker="o",
            alpha=0.7,
            label=f"{algorithm} - Seed {seed}"
        )


plt.xlabel("Target Environment Interactions")
plt.ylabel("Target Evaluation Return")

plt.title(
    "RQ1: Individual Seed Learning Curves"
)

plt.legend(
    fontsize=8,
    ncol=2
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "rq1_individual_seeds.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 3
# IMPROVEMENT FROM INITIAL PERFORMANCE
# ============================================================

improvement_summary = (
    df.groupby(
        ["Algorithm", "Target_Interactions"]
    )["Improvement_From_Initial"]
    .agg(["mean", "std"])
    .reset_index()
)


plt.figure(figsize=(9, 6))

for algorithm in results.keys():

    temp = improvement_summary[
        improvement_summary["Algorithm"] == algorithm
    ]

    x = temp["Target_Interactions"].to_numpy()
    mean = temp["mean"].to_numpy()
    std = temp["std"].to_numpy()

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=algorithm
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.15
    )


plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Target Environment Interactions")

plt.ylabel(
    "Improvement From Initial Target Return"
)

plt.title(
    "RQ1: Improvement From Initial Performance"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "rq1_improvement_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 4
# AREA UNDER LEARNING CURVE (AUC)
# ============================================================

auc_rows = []

for algorithm, seeds in results.items():

    for seed, returns in seeds.items():

        auc = np.trapz(
            returns,
            interactions
        )

        auc_rows.append([
            algorithm,
            seed,
            auc
        ])


auc_df = pd.DataFrame(
    auc_rows,
    columns=[
        "Algorithm",
        "Seed",
        "AUC"
    ]
)

auc_df.to_csv(
    "rq1_auc.csv",
    index=False
)

print("\nAUC results:")
print(auc_df)


auc_summary = (
    auc_df.groupby("Algorithm")["AUC"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\nAUC summary:")
print(auc_summary)


plt.figure(figsize=(8, 6))

plt.bar(
    auc_summary["Algorithm"],
    auc_summary["mean"],
    yerr=auc_summary["std"],
    capsize=6
)

plt.xlabel("Algorithm")

plt.ylabel(
    "Area Under Learning Curve (AUC)"
)

plt.title(
    "RQ1: Sample Efficiency Across 0-500 Target Interactions"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "rq1_auc_comparison.png",
    dpi=300
)

plt.show()


print("\n======================================")
print("RQ1 ANALYSIS COMPLETE")
print("======================================")

print("\nFiles created:")

print("1. rq1_results.csv")
print("2. rq1_summary.csv")
print("3. rq1_auc.csv")
print("4. rq1_mean_learning_curve.png")
print("5. rq1_individual_seeds.png")
print("6. rq1_improvement_curve.png")
print("7. rq1_auc_comparison.png")