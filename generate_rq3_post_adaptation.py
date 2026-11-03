import os
import re
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "experiments_ea_ntc_10k_96"
OUT = os.path.join(ROOT, "RQ3_POST_ADAPTATION_5000_10000")
os.makedirs(OUT, exist_ok=True)

TAG = "test/target_normalized_score"
EXPECTED = [5000, 6000, 7000, 8000, 9000, 10000]

records = []
seen = set()

for current, dirs, files in os.walk(ROOT):
    for filename in files:
        if not filename.startswith("events.out.tfevents"):
            continue

        path = os.path.join(current, filename)

        # The run directory is the parent of the TensorBoard folder.
        run_dir = os.path.dirname(current)
        run_name = os.path.basename(run_dir)

        match = re.match(
            r"(NORMALIZED|UNNORMALIZED)_(ant|halfcheetah|hopper|walker2d)"
            r"_shift_([0-9.]+)_seed_([0-2])$",
            run_name,
            re.IGNORECASE
        )

        if not match:
            print("UNRECOGNIZED RUN:", path)
            continue

        variant, env, shift, seed = match.groups()
        key = (variant, env, float(shift), int(seed))

        if key in seen:
            raise ValueError(f"Duplicate run found: {key}")
        seen.add(key)

        ea = EventAccumulator(path)
        ea.Reload()

        if TAG not in ea.Tags().get("scalars", []):
            raise ValueError(f"Missing tag: {TAG} in {path}")

        events = ea.Scalars(TAG)

        selected = [
            e for e in events
            if 5000 <= e.step <= 10000
        ]

        step_map = {}
        for e in selected:
            if e.step in step_map:
                raise ValueError(f"Duplicate step {e.step}: {key}")
            step_map[e.step] = e.value

        if sorted(step_map) != EXPECTED:
            raise ValueError(
                f"Incomplete interval for {key}: {sorted(step_map)}"
            )

        x = np.array(EXPECTED, dtype=float)
        y = np.array([step_map[s] for s in EXPECTED])

        # Time-averaged normalized AUC.
        # Divide integrated area by the 5,000-step interval.
        auc = np.trapz(y, x) / (10000 - 5000)

        records.append({
            "environment": env,
            "shift": float(shift),
            "seed": int(seed),
            "method": variant,
            "AUC": auc,
            "final_score": y[-1]
        })

df = pd.DataFrame(records)

expected_runs = 4 * 4 * 3 * 2

if len(df) != expected_runs:
    raise ValueError(
        f"Expected {expected_runs} runs, found {len(df)}"
    )

df.to_csv(
    os.path.join(OUT, "RQ3_seed_level_AUC_5000_10000.csv"),
    index=False
)

# Mean and sample standard deviation over three seeds.
summary = (
    df.groupby(["environment", "shift", "method"])["AUC"]
      .agg(["mean", "std"])
      .reset_index()
)

summary.to_csv(
    os.path.join(OUT, "RQ3_AUC_mean_std_5000_10000.csv"),
    index=False
)

# Pivot into paper-table format.
paper = summary.pivot(
    index=["environment", "shift"],
    columns="method",
    values=["mean", "std"]
)

print("\n========== RQ3 POST-ADAPTATION AUC ==========")
print(summary.to_string(index=False))

# Paired comparison between variants using the same seeds.
wide = df.pivot(
    index=["environment", "shift", "seed"],
    columns="method",
    values="AUC"
).reset_index()

wide["Normalized_minus_Mean"] = (
    wide["NORMALIZED"] - wide["UNNORMALIZED"]
)

wide.to_csv(
    os.path.join(OUT, "RQ3_paired_AUC_5000_10000.csv"),
    index=False
)

comparison = (
    wide.groupby(["environment", "shift"])
    ["Normalized_minus_Mean"]
    .agg(["mean", "std"])
    .reset_index()
)

comparison.to_csv(
    os.path.join(OUT, "RQ3_paired_comparison_5000_10000.csv"),
    index=False
)

print("\n========== NORMALIZED VS MEAN ==========")
print(comparison.to_string(index=False))

# Generate LaTeX table.
rows = []

for env in ["ant", "halfcheetah", "hopper", "walker2d"]:
    for shift in [0.1, 0.5, 2.0, 5.0]:
        subset = summary[
            (summary.environment == env) &
            (summary["shift"] == shift)
        ]

        values = {
            r["method"]: (r["mean"], r["std"])
            for _, r in subset.iterrows()
        }

        best = max(v[0] for v in values.values())

        cells = []
        for method in ["NORMALIZED", "UNNORMALIZED"]:
            mean, std = values[method]
            cell = f"{mean:.3f} $\\pm$ {std:.3f}"

            if np.isclose(mean, best):
                cell = r"\textbf{" + cell + "}"

            cells.append(cell)

        rows.append(
            f"{env.title()} & {shift:g} & "
            + " & ".join(cells)
            + r" \\"
        )

latex = r"""
\begin{table*}[ht]
\centering
\caption{RQ3: Time-averaged normalized post-adaptation
AUC between 5,000 and 10,000 training steps.
Values are mean $\pm$ standard deviation across three seeds.
Bold indicates the highest mean within each condition.}
\label{tab:rq3_post_adaptation}
\begin{tabular}{llcc}
\hline
Environment & Shift & EA-NTC Normalized & EA-NTC Mean \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""

with open(
    os.path.join(OUT, "RQ3_POST_ADAPTATION_TABLE.tex"),
    "w",
    encoding="utf-8"
) as f:
    f.write(latex)

print("\n========== COMPLETED ==========")
print("Verified runs:", len(df))
print("Output directory:", OUT)
print("LaTeX table: RQ3_POST_ADAPTATION_TABLE.tex")
