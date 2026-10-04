import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DIAGNOSTICS
# ============================================================

df = pd.read_csv(
    "rq3_analysis/rq3_diagnostics.csv"
)

# BC-PAR only
df = df[df["algorithm"] == "BC_PAR"].copy()

# Diagnostics of interest
metrics = [
    "train/distance",
    "train/src_reward",
    "train/encoder_loss"
]

df = df[df["metric"].isin(metrics)].copy()

# Make sure numeric columns are numeric
df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"])
df["step"] = pd.to_numeric(df["step"])
df["value"] = pd.to_numeric(df["value"])


# ============================================================
# 2. CHECK DATA
# ============================================================

print("\nRows used:")
print(len(df))

print("\nCounts by metric:")
print(df["metric"].value_counts())

print("\nSteps:")
print(sorted(df["step"].unique()))


# ============================================================
# 3. PER-SEED MEAN DIAGNOSTICS
#    Mean over steps 5000-10000
# ============================================================

mean_by_seed = (
    df.groupby(
        [
            "environment",
            "shift",
            "seed",
            "metric"
        ]
    )["value"]
    .mean()
    .reset_index()
)

mean_wide = (
    mean_by_seed
    .pivot(
        index=[
            "environment",
            "shift",
            "seed"
        ],
        columns="metric",
        values="value"
    )
    .reset_index()
)

mean_wide = mean_wide.rename(
    columns={
        "train/distance": "mean_distance",
        "train/src_reward": "mean_src_reward",
        "train/encoder_loss": "mean_encoder_loss"
    }
)


# ============================================================
# 4. FINAL DIAGNOSTICS AT STEP 10000
# ============================================================

final_df = df[df["step"] == 10000].copy()

final_wide = (
    final_df
    .pivot(
        index=[
            "environment",
            "shift",
            "seed"
        ],
        columns="metric",
        values="value"
    )
    .reset_index()
)

final_wide = final_wide.rename(
    columns={
        "train/distance": "final_distance",
        "train/src_reward": "final_src_reward",
        "train/encoder_loss": "final_encoder_loss"
    }
)


# ============================================================
# 5. MERGE PER-SEED RESULTS
# ============================================================

per_seed = pd.merge(
    mean_wide,
    final_wide,
    on=[
        "environment",
        "shift",
        "seed"
    ],
    how="outer"
)

per_seed = per_seed.sort_values(
    [
        "environment",
        "shift",
        "seed"
    ]
)

per_seed.to_csv(
    "rq3_analysis/bcpar_diagnostics_by_seed.csv",
    index=False
)


# ============================================================
# 6. SUMMARIZE ACROSS THE THREE SEEDS
# ============================================================

diagnostic_columns = [
    "mean_distance",
    "final_distance",
    "mean_src_reward",
    "final_src_reward",
    "mean_encoder_loss",
    "final_encoder_loss"
]

summary_rows = []

for (environment, shift), group in per_seed.groupby(
    ["environment", "shift"]
):

    row = {
        "environment": environment,
        "shift": shift,
        "n_seeds": group["seed"].nunique()
    }

    for column in diagnostic_columns:

        values = group[column].dropna()

        row[column + "_mean"] = values.mean()
        row[column + "_sd"] = values.std(ddof=1)

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    [
        "environment",
        "shift"
    ]
)

summary.to_csv(
    "rq3_analysis/bcpar_diagnostics_summary.csv",
    index=False
)


# ============================================================
# 7. PRINT MAIN RESULTS
# ============================================================

print("\n================================================")
print("BC-PAR DISTANCE BY ENVIRONMENT AND SHIFT")
print("================================================")

print(
    summary[
        [
            "environment",
            "shift",
            "mean_distance_mean",
            "mean_distance_sd",
            "final_distance_mean",
            "final_distance_sd"
        ]
    ].to_string(index=False)
)


print("\n================================================")
print("BC-PAR SOURCE REWARD BY ENVIRONMENT AND SHIFT")
print("================================================")

print(
    summary[
        [
            "environment",
            "shift",
            "mean_src_reward_mean",
            "mean_src_reward_sd",
            "final_src_reward_mean",
            "final_src_reward_sd"
        ]
    ].to_string(index=False)
)


print("\n================================================")
print("BC-PAR ENCODER LOSS BY ENVIRONMENT AND SHIFT")
print("================================================")

print(
    summary[
        [
            "environment",
            "shift",
            "mean_encoder_loss_mean",
            "mean_encoder_loss_sd",
            "final_encoder_loss_mean",
            "final_encoder_loss_sd"
        ]
    ].to_string(index=False)
)


print("\nSaved:")
print("rq3_analysis/bcpar_diagnostics_by_seed.csv")
print("rq3_analysis/bcpar_diagnostics_summary.csv")