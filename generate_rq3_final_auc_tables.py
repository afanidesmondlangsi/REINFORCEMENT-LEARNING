import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\LANGS\Documents\off-dynamics-rl")
DATA = ROOT / "matched_final_5000" / "final_three_RQ_analysis" / "all_learning_curve_records.csv"
OUT = ROOT / "matched_final_5000" / "RQ3_FINAL_ADAPTATION" / "RQ3_AUC_TABLES"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

required = [
    "environment", "shift", "seed", "method",
    "checkpoint", "target_return", "step"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

methods = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

environments = ["ant", "halfcheetah", "hopper", "walker2d"]
shifts = [0.1, 0.5, 2.0, 5.0]
seeds = [0, 1, 2]
checkpoints = [1000, 2000, 3000, 4000, 5000]

df["step"] = pd.to_numeric(df["step"])
df["target_return"] = pd.to_numeric(df["target_return"])
df["shift"] = pd.to_numeric(df["shift"])

df = df[df["step"].isin(checkpoints)].copy()

# Validate all 96 experimental groups.
expected = {
    (e, s, seed, m)
    for e in environments
    for s in shifts
    for seed in seeds
    for m in methods
}

actual = set()
auc_records = []

for (env, shift, seed, method), group in df.groupby(
    ["environment", "shift", "seed", "method"]
):
    key = (env, float(shift), int(seed), method)

    if key in actual:
        raise ValueError(f"Duplicate experimental group: {key}")

    actual.add(key)

    group = group.sort_values("step")

    if group["step"].tolist() != checkpoints:
        raise ValueError(f"Incorrect checkpoints for {key}")

    if group["target_return"].isna().any():
        raise ValueError(f"Missing return values for {key}")

    # Cumulative AUC from 1,000 to 5,000 steps.
    auc = np.trapz(
        group["target_return"].to_numpy(),
        group["step"].to_numpy()
    )

    auc_records.append({
        "environment": env,
        "shift": float(shift),
        "seed": int(seed),
        "method": method,
        "AUC": float(auc)
    })

missing_groups = expected - actual
unexpected_groups = actual - expected

if missing_groups or unexpected_groups:
    raise ValueError(
        f"Missing groups: {len(missing_groups)}; "
        f"Unexpected groups: {len(unexpected_groups)}"
    )

auc_df = pd.DataFrame(auc_records)

auc_df.to_csv(OUT / "RQ3_seed_level_AUC.csv", index=False)

# Calculate mean and sample standard deviation.
summary = (
    auc_df.groupby(
        ["environment", "shift", "method"]
    )["AUC"]
    .agg(["mean", "std"])
    .reset_index()
)

summary.to_csv(OUT / "RQ3_AUC_mean_std.csv", index=False)

# Table 1: Absolute AUC.
method_labels = {
    "SAC_TARGET_ONLY": "SAC Target",
    "BC_SAC": "BC-SAC",
    "BC_PAR": "BC-PAR",
    "RLPD": "RLPD",
    "EA_NTC_NORMALIZED": "EA-NTC Norm.",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean"
}

def fmt(mean, sd):
    return f"{mean:,.2f} $\\pm$ {sd:,.2f}"

lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\caption{RQ3: Cumulative target-learning AUC across four "
    r"MuJoCo environments and four friction shifts. Results are "
    r"reported as mean $\pm$ standard deviation over three seeds.}",
    r"\label{tab:rq3_absolute_auc}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{3pt}",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\begin{adjustbox}{max width=\textwidth}",
    r"\begin{tabular}{llrrrrrr}",
    r"\toprule",
    r"\textbf{Environment} & \textbf{Shift} & "
    r"\textbf{SAC Target} & \textbf{BC-SAC} & "
    r"\textbf{BC-PAR} & \textbf{RLPD} & "
    r"\textbf{EA-NTC Norm.} & \textbf{EA-NTC Mean} \\",
    r"\midrule"
]

for env in environments:
    for shift in shifts:
        subset = summary[
            (summary.environment == env) &
            (summary["shift"] == shift)
        ]

        values = {}
        for method in methods:
            row = subset[subset.method == method]
            if len(row) != 1:
                raise ValueError(f"Missing result: {env}, {shift}, {method}")
            values[method] = (row.iloc[0]["mean"], row.iloc[0]["std"])

        best = max(values, key=lambda m: values[m][0])

        cells = []
        for method in methods:
            mean, sd = values[method]
            cell = fmt(mean, sd)
            if method == best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        lines.append(
            f"{env.title()} & {shift:g} & " +
            " & ".join(cells) + r" \\"
        )

    if env != environments[-1]:
        lines.append(r"\midrule")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{adjustbox}",
    r"\begin{flushleft}",
    r"\footnotesize \textit{Note:} AUC is calculated using "
    r"the trapezoidal rule over target-training steps "
    r"1,000--5,000. Values represent mean $\pm$ sample "
    r"standard deviation across seeds 0, 1 and 2. "
    r"Bold indicates the highest mean AUC in each "
    r"environment--shift condition.",
    r"\end{flushleft}",
    r"\end{table*}"
]

(OUT / "RQ3_absolute_AUC_table.tex").write_text(
    "\n".join(lines), encoding="utf-8"
)

# Table 2: Paired AUC advantage relative to SAC Target-Only.
paired = auc_df.pivot_table(
    index=["environment", "shift", "seed"],
    columns="method",
    values="AUC"
).reset_index()

for method in methods:
    if method == "SAC_TARGET_ONLY":
        paired[method + "_DELTA"] = 0.0
    else:
        paired[method + "_DELTA"] = (
            paired[method] - paired["SAC_TARGET_ONLY"]
        )

delta_methods = methods[1:]

delta_records = []

for method in delta_methods:
    for (env, shift), group in paired.groupby(
        ["environment", "shift"]
    ):
        vals = group[method + "_DELTA"].to_numpy()

        delta_records.append({
            "environment": env,
            "shift": shift,
            "method": method,
            "mean_delta_AUC": np.mean(vals),
            "std_delta_AUC": np.std(vals, ddof=1),
            "positive_seeds": int(np.sum(vals > 0)),
            "negative_seeds": int(np.sum(vals < 0)),
            "neutral_seeds": int(np.sum(vals == 0))
        })

delta_df = pd.DataFrame(delta_records)
delta_df.to_csv(OUT / "RQ3_paired_AUC_advantage.csv", index=False)

delta_lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\caption{RQ3: Paired cumulative AUC advantage relative "
    r"to SAC Target-Only across environments and friction shifts.}",
    r"\label{tab:rq3_paired_auc}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{adjustbox}{max width=\textwidth}",
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"\textbf{Environment} & \textbf{Shift} & "
    r"\textbf{BC-SAC} & \textbf{BC-PAR} & "
    r"\textbf{RLPD} & \textbf{EA-NTC Norm.} & "
    r"\textbf{EA-NTC Mean} \\",
    r"\midrule"
]

for env in environments:
    for shift in shifts:
        subset = delta_df[
            (delta_df.environment == env) &
            (delta_df["shift"] == shift)
        ]

        cells = []
        for method in delta_methods:
            row = subset[subset.method == method].iloc[0]
            value = fmt(
                row["mean_delta_AUC"],
                row["std_delta_AUC"]
            )
            cells.append(value)

        delta_lines.append(
            f"{env.title()} & {shift:g} & " +
            " & ".join(cells) + r" \\"
        )

    if env != environments[-1]:
        delta_lines.append(r"\midrule")

delta_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{adjustbox}",
    r"\begin{flushleft}",
    r"\footnotesize \textit{Note:} Each seed-level AUC "
    r"difference is calculated against SAC Target-Only "
    r"under the same environment and friction shift. "
    r"Positive values indicate improved cumulative "
    r"target performance; negative values indicate "
    r"reduced performance. Values represent mean "
    r"$\pm$ sample standard deviation across three seeds.",
    r"\end{flushleft}",
    r"\end{table*}"
]

(OUT / "RQ3_paired_AUC_table.tex").write_text(
    "\n".join(delta_lines), encoding="utf-8"
)

print("\n========== RQ3 FINAL AUC SUMMARY ==========")
print(f"Experimental groups: {len(auc_df)}")
print(f"Environments: {len(environments)}")
print(f"Friction shifts: {len(shifts)}")
print(f"Algorithms: {len(methods)}")
print(f"Seeds per condition: {len(seeds)}")
print("AUC interval: 1,000 to 5,000 steps")

print("\n========== OVERALL MEAN AUC ==========")
overall = (
    auc_df.groupby("method")["AUC"]
    .agg(["mean", "std"])
    .reset_index()
)
print(overall.to_string(index=False))

print("\n========== PAIRED AUC ADVANTAGE ==========")
overall_delta = (
    delta_df.groupby("method")
    .agg(
        mean_delta=("mean_delta_AUC", "mean"),
        positive=("positive_seeds", "sum"),
        negative=("negative_seeds", "sum")
    )
    .reset_index()
)
print(overall_delta.to_string(index=False))

print("\n========== GENERATED FILES ==========")
for f in sorted(OUT.glob("*")):
    print(f.name)

print(f"\nOutput directory: {OUT}")
