import os
import numpy as np
import pandas as pd

BASE = "matched_final_5000"
DATA = r'C:\Users\LANGS\Documents\off-dynamics-rl\matched_final_5000\final_three_RQ_analysis\all_learning_curve_records.csv'

OUT = os.path.join(
    BASE, "RQ2_FINAL_AUC_AUDIT"
)
os.makedirs(OUT, exist_ok=True)

EXPECTED_METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

EXPECTED_ENVS = [
    "ant", "halfcheetah", "hopper", "walker2d"
]

EXPECTED_SHIFTS = [0.1, 0.5, 2.0, 5.0]
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_STEPS = [1000, 2000, 3000, 4000, 5000]

print("\n========== RQ2 FINAL AUC AUDIT ==========")

df = pd.read_csv(DATA)

required = [
    "environment", "shift", "seed",
    "method", "target_return", "step"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["environment"] = df["environment"].astype(str).str.lower()
df["method"] = df["method"].astype(str).str.upper()
df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
df["step"] = pd.to_numeric(df["step"], errors="raise").astype(int)
df["target_return"] = pd.to_numeric(
    df["target_return"], errors="raise"
)

print("Records:", len(df))
print("Methods:", sorted(df["method"].unique()))
print("Environments:", sorted(df["environment"].unique()))
print("Seeds:", sorted(df["seed"].unique()))
print("Shifts:", sorted(df["shift"].unique()))

# Check duplicate evaluation records.
key = [
    "environment", "shift", "seed",
    "method", "step"
]

duplicates = df[df.duplicated(key, keep=False)]

if not duplicates.empty:
    duplicates.to_csv(
        os.path.join(OUT, "duplicate_records.csv"),
        index=False
    )
    raise ValueError(
        "Duplicate evaluation records found. "
        "Inspect duplicate_records.csv."
    )

# Check the expected experimental design.
if set(df["method"].unique()) != set(EXPECTED_METHODS):
    raise ValueError("Unexpected or missing algorithms.")

if set(df["environment"].unique()) != set(EXPECTED_ENVS):
    raise ValueError("Unexpected or missing environments.")

if set(df["seed"].unique()) != set(EXPECTED_SEEDS):
    raise ValueError("Unexpected or missing seeds.")

if set(df["shift"].unique()) != set(EXPECTED_SHIFTS):
    raise ValueError("Unexpected or missing friction shifts.")

# Calculate AUC independently for every seed.
auc_records = []

group_cols = [
    "environment", "shift", "seed", "method"
]

for keys, group in df.groupby(group_cols):

    env, shift, seed, method = keys

    group = group.sort_values("step")

    steps = group["step"].to_numpy()
    returns = group["target_return"].to_numpy()

    if list(steps) != EXPECTED_STEPS:
        raise ValueError(
            f"Incorrect checkpoints: {keys}, {list(steps)}"
        )

    auc = np.trapz(returns, steps)

    auc_records.append({
        "environment": env,
        "shift": shift,
        "seed": seed,
        "method": method,
        "AUC": auc
    })

auc_df = pd.DataFrame(auc_records)

expected_rows = 4 * 4 * 3 * 6

if len(auc_df) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} AUC records, "
        f"found {len(auc_df)}"
    )

# Pair every method against SAC using identical conditions and seeds.
baseline = auc_df[
    auc_df["method"] == "SAC_TARGET_ONLY"
][[
    "environment", "shift", "seed", "AUC"
]].rename(columns={"AUC": "SAC_AUC"})

paired = auc_df.merge(
    baseline,
    on=["environment", "shift", "seed"],
    how="left",
    validate="many_to_one"
)

paired["delta_AUC"] = paired["AUC"] - paired["SAC_AUC"]

if paired["SAC_AUC"].isna().any():
    raise ValueError("Missing paired SAC baseline.")

paired.to_csv(
    os.path.join(OUT, "RQ2_paired_seed_AUC.csv"),
    index=False
)

# Average the three paired seed differences per condition.
condition = (
    paired.groupby(
        ["environment", "shift", "method"]
    )
    .agg(
        mean_AUC=("AUC", "mean"),
        mean_SAC_AUC=("SAC_AUC", "mean"),
        mean_delta_AUC=("delta_AUC", "mean"),
        positive_seeds=("delta_AUC", lambda x: (x > 0).sum()),
        negative_seeds=("delta_AUC", lambda x: (x < 0).sum()),
        neutral_seeds=("delta_AUC", lambda x: (x == 0).sum()),
        seed_count=("seed", "nunique")
    )
    .reset_index()
)

condition.to_csv(
    os.path.join(OUT, "RQ2_condition_level_AUC.csv"),
    index=False
)

# Condition-level positive/negative transfer.
methods = [
    m for m in EXPECTED_METHODS
    if m != "SAC_TARGET_ONLY"
]

summary = []

for method in methods:

    sub = condition[condition["method"] == method]

    positive = int((sub["mean_delta_AUC"] > 0).sum())
    negative = int((sub["mean_delta_AUC"] < 0).sum())
    neutral = int((sub["mean_delta_AUC"] == 0).sum())

    summary.append({
        "method": method,
        "mean_delta_AUC": sub["mean_delta_AUC"].mean(),
        "positive_conditions": positive,
        "negative_conditions": negative,
        "neutral_conditions": neutral,
        "positive_pct": positive / 16 * 100,
        "negative_pct": negative / 16 * 100
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    os.path.join(OUT, "RQ2_FINAL_TRANSFER_FREQUENCY.csv"),
    index=False
)

# Also calculate seed-level frequency separately.
seed_summary = (
    paired[paired["method"] != "SAC_TARGET_ONLY"]
    .groupby("method")
    .agg(
        positive_seed_cases=("delta_AUC", lambda x: (x > 0).sum()),
        negative_seed_cases=("delta_AUC", lambda x: (x < 0).sum()),
        neutral_seed_cases=("delta_AUC", lambda x: (x == 0).sum()),
        mean_seed_delta=("delta_AUC", "mean"),
        total_seed_cases=("delta_AUC", "size")
    )
    .reset_index()
)

seed_summary.to_csv(
    os.path.join(OUT, "RQ2_SEED_LEVEL_FREQUENCY.csv"),
    index=False
)

print("\n========== CONDITION-LEVEL RESULTS ==========")
print(summary_df.to_string(index=False))

print("\n========== SEED-LEVEL RESULTS ==========")
print(seed_summary.to_string(index=False))

print("\n========== ENVIRONMENT-LEVEL RESULTS ==========")

environment_summary = (
    condition[condition["method"] != "SAC_TARGET_ONLY"]
    .groupby(["method", "environment"])
    .agg(
        mean_delta_AUC=("mean_delta_AUC", "mean"),
        positive_conditions=(
            "mean_delta_AUC", lambda x: (x > 0).sum()
        ),
        negative_conditions=(
            "mean_delta_AUC", lambda x: (x < 0).sum()
        )
    )
    .reset_index()
)

environment_summary.to_csv(
    os.path.join(OUT, "RQ2_ENVIRONMENT_SUMMARY.csv"),
    index=False
)

print(environment_summary.to_string(index=False))

print("\n========== AUDIT COMPLETED ==========")
print("Output directory:", os.path.abspath(OUT))
print("AUC records:", len(auc_df))
print("Paired records:", len(paired))
print("Conditions per algorithm:", len(condition) // 6)

