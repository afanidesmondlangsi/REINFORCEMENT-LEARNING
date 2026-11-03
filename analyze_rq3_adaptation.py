import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = r"C:\Users\LANGS\Documents\off-dynamics-rl"
DATA = os.path.join(
    PROJECT,
    "matched_final_5000",
    "final_three_RQ_analysis",
    "all_learning_curve_records.csv"
)
OUT = os.path.join(
    PROJECT, "matched_final_5000", "RQ3_FINAL_ADAPTATION"
)

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

df["method"] = df["method"].str.upper()
df["environment"] = df["environment"].str.lower()
df["shift"] = df["shift"].astype(float)
df["seed"] = df["seed"].astype(int)
df["step"] = df["step"].astype(int)
df["target_return"] = pd.to_numeric(
    df["target_return"], errors="raise"
)

methods = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
    "SAC_TARGET_ONLY"
]

df = df[df["method"].isin(methods)].copy()

# Validate each experimental run
expected_steps = [1000, 2000, 3000, 4000, 5000]

records = []

for keys, group in df.groupby(
    ["environment", "shift", "method", "seed"]
):
    env, shift, method, seed = keys
    group = group.sort_values("step")

    if group["step"].tolist() != expected_steps:
        raise ValueError(
            f"Incorrect checkpoints: {keys}"
        )

    if group["step"].duplicated().any():
        raise ValueError(
            f"Duplicate checkpoints: {keys}"
        )

    auc = np.trapz(
        group["target_return"].to_numpy(),
        group["step"].to_numpy()
    )

    records.append({
        "environment": env,
        "shift": shift,
        "method": method,
        "seed": seed,
        "AUC": auc,
        "final_return": group.loc[
            group["step"] == 5000,
            "target_return"
        ].iloc[0]
    })

runs = pd.DataFrame(records)

# Pair all methods by environment, shift and seed
wide_auc = runs.pivot(
    index=["environment", "shift", "seed"],
    columns="method",
    values="AUC"
).reset_index()

wide_final = runs.pivot(
    index=["environment", "shift", "seed"],
    columns="method",
    values="final_return"
).reset_index()

def add_differences(data):
    data = data.copy()

    data["delta_AUC_Norm_Mean"] = (
        data["EA_NTC_NORMALIZED"]
        - data["EA_NTC_UNNORMALIZED"]
    )

    data["delta_AUC_Norm_SAC"] = (
        data["EA_NTC_NORMALIZED"]
        - data["SAC_TARGET_ONLY"]
    )

    data["delta_AUC_Mean_SAC"] = (
        data["EA_NTC_UNNORMALIZED"]
        - data["SAC_TARGET_ONLY"]
    )

    return data

wide_auc = add_differences(wide_auc)

wide_final["delta_Final_Norm_Mean"] = (
    wide_final["EA_NTC_NORMALIZED"]
    - wide_final["EA_NTC_UNNORMALIZED"]
)

wide_final["delta_Final_Norm_SAC"] = (
    wide_final["EA_NTC_NORMALIZED"]
    - wide_final["SAC_TARGET_ONLY"]
)

wide_final["delta_Final_Mean_SAC"] = (
    wide_final["EA_NTC_UNNORMALIZED"]
    - wide_final["SAC_TARGET_ONLY"]
)

wide_auc.to_csv(
    os.path.join(OUT, "RQ3_paired_seed_AUC.csv"),
    index=False
)

wide_final.to_csv(
    os.path.join(OUT, "RQ3_paired_seed_final_return.csv"),
    index=False
)

# Summarize all 16 environment-shift conditions
condition_auc = wide_auc.groupby(
    ["environment", "shift"]
).agg(
    mean_AUC_difference=("delta_AUC_Norm_Mean", "mean"),
    std_AUC_difference=("delta_AUC_Norm_Mean", "std"),
    normalized_positive_seeds=(
        "delta_AUC_Norm_Mean",
        lambda x: int((x > 0).sum())
    ),
    mean_positive_seeds=(
        "delta_AUC_Norm_Mean",
        lambda x: int((x < 0).sum())
    )
).reset_index()

condition_final = wide_final.groupby(
    ["environment", "shift"]
).agg(
    mean_final_difference=("delta_Final_Norm_Mean", "mean"),
    std_final_difference=("delta_Final_Norm_Mean", "std")
).reset_index()

condition = condition_auc.merge(
    condition_final,
    on=["environment", "shift"]
)

condition.to_csv(
    os.path.join(OUT, "RQ3_condition_level_comparison.csv"),
    index=False
)

# Environment-level summary
environment_summary = wide_auc.groupby(
    "environment"
).agg(
    mean_AUC_difference=("delta_AUC_Norm_Mean", "mean"),
    std_AUC_difference=("delta_AUC_Norm_Mean", "std"),
    normalized_wins=("delta_AUC_Norm_Mean",
                     lambda x: int((x > 0).sum())),
    mean_wins=("delta_AUC_Norm_Mean",
               lambda x: int((x < 0).sum()))
).reset_index()

environment_summary.to_csv(
    os.path.join(OUT, "RQ3_environment_summary.csv"),
    index=False
)

# Overall comparison
overall = pd.DataFrame([
    {
        "Comparison": "Normalized vs Mean",
        "Mean_AUC_difference":
            wide_auc["delta_AUC_Norm_Mean"].mean(),
        "Normalized_wins":
            int((wide_auc["delta_AUC_Norm_Mean"] > 0).sum()),
        "Mean_wins":
            int((wide_auc["delta_AUC_Norm_Mean"] < 0).sum()),
        "Ties":
            int((wide_auc["delta_AUC_Norm_Mean"] == 0).sum())
    },
    {
        "Comparison": "Normalized vs SAC",
        "Mean_AUC_difference":
            wide_auc["delta_AUC_Norm_SAC"].mean(),
        "Normalized_wins":
            int((wide_auc["delta_AUC_Norm_SAC"] > 0).sum()),
        "Mean_wins":
            int((wide_auc["delta_AUC_Norm_SAC"] < 0).sum()),
        "Ties":
            int((wide_auc["delta_AUC_Norm_SAC"] == 0).sum())
    },
    {
        "Comparison": "Mean vs SAC",
        "Mean_AUC_difference":
            wide_auc["delta_AUC_Mean_SAC"].mean(),
        "Normalized_wins":
            int((wide_auc["delta_AUC_Mean_SAC"] > 0).sum()),
        "Mean_wins":
            int((wide_auc["delta_AUC_Mean_SAC"] < 0).sum()),
        "Ties":
            int((wide_auc["delta_AUC_Mean_SAC"] == 0).sum())
    }
])

overall.to_csv(
    os.path.join(OUT, "RQ3_overall_summary.csv"),
    index=False
)

# Plot 1: Paired AUC difference across shifts
fig, axes = plt.subplots(
    2, 2, figsize=(14, 10), sharex=True
)

for ax, env in zip(axes.flat,
                   ["ant", "halfcheetah", "hopper", "walker2d"]):

    sub = wide_auc[
        wide_auc["environment"] == env
    ]

    summary = sub.groupby("shift")[
        "delta_AUC_Norm_Mean"
    ].agg(["mean", "std"])

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        marker="o",
        capsize=4,
        linewidth=2
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(env.capitalize())
    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Normalized - Mean AUC")
    ax.grid(alpha=0.25)

fig.suptitle(
    "RQ3: EA-NTC Paired AUC Advantage Across Friction Shifts"
)
fig.tight_layout()

fig.savefig(
    os.path.join(OUT, "RQ3_paired_AUC_by_environment.png"),
    dpi=300
)
fig.savefig(
    os.path.join(OUT, "RQ3_paired_AUC_by_environment.pdf")
)
plt.close(fig)

# Plot 2: Final target return difference
fig, axes = plt.subplots(
    2, 2, figsize=(14, 10), sharex=True
)

for ax, env in zip(axes.flat,
                   ["ant", "halfcheetah", "hopper", "walker2d"]):

    sub = wide_final[
        wide_final["environment"] == env
    ]

    summary = sub.groupby("shift")[
        "delta_Final_Norm_Mean"
    ].agg(["mean", "std"])

    ax.errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        marker="s",
        capsize=4,
        linewidth=2
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(env.capitalize())
    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Normalized - Mean final return")
    ax.grid(alpha=0.25)

fig.suptitle(
    "RQ3: Final Target Return Difference at 5,000 Steps"
)
fig.tight_layout()

fig.savefig(
    os.path.join(OUT, "RQ3_final_return_by_environment.png"),
    dpi=300
)
fig.savefig(
    os.path.join(OUT, "RQ3_final_return_by_environment.pdf")
)
plt.close(fig)

print("\n========== RQ3 OVERALL RESULTS ==========")
print(overall.to_string(index=False))

print("\n========== RQ3 ENVIRONMENT SUMMARY ==========")
print(environment_summary.to_string(index=False))

print("\n========== RQ3 CONDITION SUMMARY ==========")
print(condition.to_string(index=False))

print("\n========== RQ3 OUTPUT FILES ==========")
for f in sorted(os.listdir(OUT)):
    print(f)

print("\nRQ3 ANALYSIS COMPLETED SUCCESSFULLY.")
print("Output:", OUT)
