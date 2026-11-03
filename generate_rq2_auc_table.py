import pandas as pd
import numpy as np
from pathlib import Path

root = Path("matched_final_5000")

data_file = root / "matched_final_288_long.csv"

out_dir = root / "final_three_RQ_analysis" / "paper_tables"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)

steps = [1000, 2000, 3000, 4000, 5000]
eval_cols = [f"eval_{s}" for s in steps]

methods = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

labels = {
    "SAC_TARGET_ONLY": "SAC Target",
    "BC_SAC": "BC-SAC",
    "BC_PAR": "BC-PAR",
    "RLPD": "RLPD",
    "EA_NTC_NORMALIZED": "EA-NTC Norm.",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean"
}

# Validate experimental records
assert len(df) == 288
assert df.groupby(
    ["environment", "shift", "seed", "method"]
).size().max() == 1

assert df[eval_cols].notna().all().all()

# Calculate seed-level cumulative AUC
df["auc_1000_5000"] = df.apply(
    lambda row: np.trapz(
        row[eval_cols].to_numpy(dtype=float),
        steps
    ),
    axis=1
)

# Average AUC across three seeds
summary = (
    df.groupby(["environment", "shift", "method"])
    ["auc_1000_5000"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

assert (summary["count"] == 3).all()
assert len(summary) == 96

summary["method"] = pd.Categorical(
    summary["method"],
    categories=methods,
    ordered=True
)

summary = summary.sort_values(
    ["environment", "shift", "method"]
)

# Save detailed AUC results
summary.to_csv(
    out_dir / "RQ2_AUC_mean_std_all_algorithms.csv",
    index=False
)

# Wide table
wide = summary.pivot(
    index=["environment", "shift"],
    columns="method",
    values="mean"
).reset_index()

wide = wide.rename(columns=labels)

wide.to_csv(
    out_dir / "RQ2_AUC_paper_table.csv",
    index=False
)

print("\n========== RQ2: MEAN CUMULATIVE AUC ==========")
print("AUC interval: 1,000 to 5,000 training steps")
print("Seeds: 0, 1, 2")
print("Total records:", len(df))

print("\n", wide.to_string(
    index=False,
    float_format=lambda x: f"{x:,.2f}"
))

# Generate LaTeX table
latex = wide.to_latex(
    index=False,
    escape=True,
    float_format=lambda x: f"{x:,.2f}",
    caption=(
        "Mean cumulative target-learning AUC across four "
        "MuJoCo environments and four friction shifts. "
        "Values are averaged over three random seeds "
        "between 1,000 and 5,000 training steps."
    ),
    label="tab:rq2_auc_all_algorithms",
    column_format="llrrrrrr"
)

with open(
    out_dir / "RQ2_AUC_table.tex",
    "w",
    encoding="utf-8"
) as f:
    f.write(latex)

print("\nFiles generated:")
print(out_dir / "RQ2_AUC_mean_std_all_algorithms.csv")
print(out_dir / "RQ2_AUC_paper_table.csv")
print(out_dir / "RQ2_AUC_table.tex")

