import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("matched_corrected_revised_5000")
SOURCE = ROOT / "matched_corrected_revised_long.csv"
OUT = ROOT / "final_analysis"
OUT.mkdir(exist_ok=True)

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

STEPS = [1000, 2000, 3000, 4000, 5000]

df = pd.read_csv(SOURCE)
records = []
curves = []
errors = []

for _, row in df.iterrows():
    event_dir = Path(str(row["path"]))

    try:
        acc = EventAccumulator(
            str(event_dir),
            size_guidance={"scalars": 0}
        )
        acc.Reload()

        tags = acc.Tags()["scalars"]

        required = [
            "test/target_return",
            "test/target_normalized_score"
        ]

        missing = [tag for tag in required if tag not in tags]

        if missing:
            raise ValueError(f"Missing scalar tags: {missing}")

        values = {}

        for tag in required:
            events = acc.Scalars(tag)
            values[tag] = {
                event.step: event.value for event in events
            }

        raw = []
        normalized = []

        for step in STEPS:
            if step not in values["test/target_return"]:
                raise ValueError(f"Missing raw return at step {step}")

            if step not in values["test/target_normalized_score"]:
                raise ValueError(
                    f"Missing normalized score at step {step}"
                )

            r = values["test/target_return"][step]
            n = values["test/target_normalized_score"][step]

            raw.append(r)
            normalized.append(n)

            curves.append({
                "environment": row["environment"],
                "shift": row["shift"],
                "seed": row["seed"],
                "method": row["method"],
                "step": step,
                "target_return": r,
                "normalized_score": n
            })

        records.append({
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "method": row["method"],
            "target_return_5000": raw[-1],
            "normalized_score_5000": normalized[-1],
            "auc_return_1000_5000": np.trapz(raw, STEPS),
            "auc_normalized_1000_5000": np.trapz(
                normalized, STEPS
            )
        })

    except Exception as e:
        errors.append({
            "method": row["method"],
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "path": str(event_dir),
            "error": str(e)
        })

metrics = pd.DataFrame(records)
curve_df = pd.DataFrame(curves)

metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
curve_df.to_csv(OUT / "learning_curves_long.csv", index=False)

if errors:
    pd.DataFrame(errors).to_csv(
        OUT / "analysis_errors.csv", index=False
    )

    print("\nERRORS FOUND:")
    print(pd.DataFrame(errors).to_string(index=False))
    raise SystemExit("Analysis incomplete. Resolve missing records.")

if len(metrics) != 90:
    raise SystemExit(
        f"Expected 90 metric records, obtained {len(metrics)}"
    )

# Mean and standard deviation across matched seeds.
metric_columns = [
    "normalized_score_5000",
    "target_return_5000",
    "auc_normalized_1000_5000",
    "auc_return_1000_5000"
]

summary = (
    metrics.groupby(
        ["environment", "shift", "method"]
    )[metric_columns]
    .agg(["mean", "std", "count"])
)

summary.to_csv(OUT / "summary_mean_std.csv")

# Separate readable mean +/- SD tables.
for metric in metric_columns:
    table = (
        metrics.groupby(
            ["environment", "shift", "method"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    table["mean_std"] = table.apply(
        lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}",
        axis=1
    )

    table.to_csv(
        OUT / f"{metric}_summary.csv",
        index=False
    )

# Generate five matched learning-curve plots.
for (env, shift), group in curve_df.groupby(
    ["environment", "shift"]
):
    plt.figure(figsize=(11, 6))

    for method in METHODS:
        subset = group[group["method"] == method]

        if subset.empty:
            continue

        stats = subset.groupby("step")[
            "normalized_score"
        ].agg(["mean", "std"])

        plt.plot(
            stats.index,
            stats["mean"],
            marker="o",
            label=method
        )

        plt.fill_between(
            stats.index,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            alpha=0.12
        )

    plt.title(
        f"{env.capitalize()} Friction {shift}: "
        "Normalized Target Performance"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Normalized Target Score")
    plt.xticks(STEPS)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        OUT / f"{env}_shift_{shift}_learning_curve.png",
        dpi=300
    )
    plt.close()

print("\n========== FINAL 5000-STEP ANALYSIS ==========")
print("Metric records:", len(metrics))
print("Algorithms:", metrics["method"].nunique())
print("Environments:", metrics["environment"].nunique())
print("Matched seeds:", sorted(metrics["seed"].unique()))

for metric in metric_columns:
    print(f"\n========== {metric.upper()} ==========")

    table = (
        metrics.groupby(
            ["environment", "shift", "method"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for _, r in table.iterrows():
        print(
            f"{r['environment']:10s} "
            f"shift={r['shift']:3.1f} "
            f"{r['method']:22s} "
            f"{r['mean']:.4f} +/- {r['std']:.4f} "
            f"(n={r['count']})"
        )

print("\nAnalysis completed successfully.")
print("Output directory:", OUT.resolve())
